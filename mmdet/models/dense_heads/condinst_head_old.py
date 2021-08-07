import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, force_fp32
from mmcv.image import tensor2imgs
from skimage import color

from mmdet.core import distance2bbox, multi_apply, multiclass_nms, reduce_mean
from ..builder import HEADS, build_loss
from .fcos_head import FCOSHead

INF = 1e8


def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    # this equation is equal to log(p_i * p_j + (1 - p_i) * (1 - p_j))
    # max is used to prevent overflow
    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)  #
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    return -log_same_prob[:, 0]


def dice_coefficient(x, target):
    """
    Dice Loss: 1 - 2 * (intersection(A, B) / (A^2 + B^2))
    :param x:
    :param target:
    :return:
    """
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss

def compute_project_term(mask_scores, gt_bitmasks):
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert isinstance(factor, int)

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )
    return tensor[:, :, :oh - 1, :ow - 1]



def get_original_image(img, img_meta):
    """

    :param img(Tensor):  the image with pading [3, h, w]
    :param img_meta(dict): information about the image
    :return: original_img(Tensor)
    """
    original_shape = img_meta["img_shape"]
    original_shape_img = img[:, :original_shape[0], :original_shape[1]]
    img_norm_cfg = img_meta["img_norm_cfg"]
    original_img = tensor2imgs(original_shape_img.unsqueeze(0), mean=img_norm_cfg["mean"], std=img_norm_cfg["std"], to_rgb=img_norm_cfg["to_rgb"])[0] #in BGR format
    original_img = torch.tensor(original_img[:, :, ::-1].copy()).permute(2, 0, 1) #to RGB tensor [c h w]
    original_img = original_img.float().to(img.device)

    return original_img

    # cv2.imwrite("show/cv_{}".format(img_meta["filename"].split("/")[-1]), original_img)
    # Image.fromarray(original_img).save("show/pil_{}".format(img_meta["filename"].split("/")[-1]))


def unfold_wo_center(x, kernel_size, dilation):
    """
    :param x: [N, C, H, W]
    :param kernel_size: k
    :param dilation:
    :return: [N, C, K^2-1, H, W]
    """
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((unfolded_x[:, :, :size // 2], unfolded_x[:, :, size // 2 + 1:]), dim=2)

    return unfolded_x


def get_image_color_similarity(image, mask, pairwise_size, pairwise_dilation):
    """
    \
    :param self:
    :param image: [1, 3, H, W]
    :param mask: [H, W]
    :param pairwise_size: k
    :param pairwise_dilation: d
    :return:[1, 8, H, W]
    """
    assert image.dim() == 4
    assert image.size(0) == 1

    unfolded_image = unfold_wo_center(
        image, kernel_size=pairwise_size, dilation=pairwise_dilation
    )

    diff = image.unsqueeze(2) - unfolded_image

    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    unfolded_weight = unfold_wo_center(
        mask.unsqueeze(0).unsqueeze(0),
        kernel_size=pairwise_size, dilation=pairwise_dilation
    )[:, 0, :, :, :]

    return similarity * unfolded_weight


@HEADS.register_module()
class CondInstBoxHead(FCOSHead):

    def forward(self, feats, top_module):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            top_module (nn.Module): Generate dynamic parameters from FCOS
                regression branch.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
                param_preds (list[Tensor]): dynamic parameters generated from \
                    each scale level, each is a 4-D-tensor, the channel number \
                    is decided by top_module.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides, top_module=top_module)
    
    def forward_single(self, x, scale, stride, top_module):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.
            top_module (nn.Module): Exteral input module. #---------------

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = \
                super(FCOSHead, self).forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        param_pred = top_module(reg_feat)
        return cls_score, bbox_pred, centerness, param_pred

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, bbox_targets, gt_inds = \
                self.get_targets(all_level_points, gt_bboxes, gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        flatten_gt_inds = torch.cat(gt_inds)
        flatten_img_inds = []
        flatten_level_inds = []
        for i, featmap_size in enumerate(featmap_sizes):
            H, W = featmap_size
            img_inds = torch.arange(num_imgs, device=bbox_preds[0].device)
            flatten_img_inds.append(img_inds.repeat_interleave(H * W))
            flatten_level_inds.append(torch.full(
                (num_imgs * H * W, ), i, device=bbox_preds[0].device).long())
        flatten_img_inds = torch.cat(flatten_img_inds)
        flatten_level_inds = torch.cat(flatten_level_inds)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        losses = dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)
        return (losses, flatten_points, flatten_level_inds, flatten_img_inds,
                flatten_gt_inds)

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, gt_inds_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        cum = 0
        for gt_inds, gt_bboxes in zip(gt_inds_list, gt_bboxes_list):
            gt_inds[gt_inds != -1] += cum
            cum += gt_bboxes.size(0)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        gt_inds_list = [gt_inds.split(num_points, 0) for gt_inds in gt_inds_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_gt_inds = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_gt_inds.append(
                torch.cat([gt_inds[i] for gt_inds in gt_inds_list]))
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_gt_inds)

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        min_area_inds[min_area == INF] = -1

        return labels, bbox_targets, min_area_inds

    def simple_test(feats, img_metas, top_module, rescale=False):
        









































@HEADS.register_module()
class CondInstSegmHead(BaseModule):

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 in_stride=8,
                 stacked_convs=2,
                 feat_channels=128,
                 loss_segm=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 norm_cfg=dict(
                     type='BN',
                     requires_grad=True),
                 init_cfg=dict(
                     type='Kaiming',
                     layer="Conv2d",
                     distribution='uniform',
                     a=1,
                     mode='fan_in',
                     nonlinearity='leaky_relu',
                     override=dict(
                         type='Kaiming',
                         name='segm_conv',
                         bias_prob=0.01))):
        super(CondInstSegmHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.in_stride = in_stride
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.loss_segm = build_loss(loss_segm)
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self):
        segm_branch = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            segm_branch.append(ConvModule(
                chn,
                self.feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=self.norm_cfg))
        self.segm_branch = nn.Sequential(*segm_branch)
        self.segm_conv = nn.Conv2d(
            self.feat_channels, self.num_classes, kernel_size=1)

    def forward(self, x):
        return self.segm_conv(self.segm_branch(x))

    @force_fp32(apply_to=('segm_pred', ))
    def loss(self, segm_pred, gt_masks, gt_labels):
        semantic_targets = self.get_targets(gt_masks, gt_labels)
        semantic_targets = semantic_targets.flatten()
        num_pos = (semantic_targets != self.num_classes).sum().float()
        num_pos = num_pos.clamp(min=1.)

        segm_pred = segm_pred.permute(0, 2, 3, 1)
        segm_pred = segm_pred.flatten(end_dim=2)
        loss_segm = self.loss_segm(
            segm_pred,
            semantic_targets,
            avg_factor=num_pos)
        return dict(loss_segm=loss_segm)

    def get_targets(self, gt_masks, gt_labels):
        semantic_targets = []
        for cur_gt_masks, cur_gt_labels in zip(gt_masks, gt_labels):
            h, w = cur_gt_masks.size()[-2:]
            areas = torch.sum(cur_gt_masks, dim=(1, 2), keepdim=True)
            areas = areas.repeat(1, h, w)
            areas[cur_gt_masks == 0] = INF
            min_areas, inds = torch.min(areas, dim=0, keepdim=True)

            cur_gt_labels = cur_gt_labels[:, None, None].repeat(1, h, w)
            per_img_targets = torch.gather(cur_gt_labels, 0, inds)
            per_img_targets[min_areas == INF] = self.num_classes
            semantic_targets.append(per_img_targets)

        stride = self.in_stride
        semantic_targets = torch.cat(semantic_targets, dim=0)
        semantic_targets = semantic_targets[:, stride//2::stride, stride//2::stride]
        return semantic_targets


@HEADS.register_module()
class CondInstMaskBranch(BaseModule):

    def __init__(self,
                 in_channels=256,
                 in_indices=[0, 1, 2],
                 strides=[8, 16, 32],
                 branch_convs=4,
                 branch_channels=128,
                 branch_out_channels=8,
                 norm_cfg=dict(
                     type='BN',
                     requires_grad=True),
                 init_cfg=dict(
                     type='Kaiming',
                     layer="Conv2d",
                     distribution='uniform',
                     a=1,
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super(CondInstMaskBranch, self).__init__(init_cfg)
        self.in_channels = in_channels
        assert len(in_indices) == len(strides)
        assert in_indices[0] == 0
        self.in_indices = in_indices
        self.strides = strides
        self.branch_convs = branch_convs
        self.branch_channels = branch_channels
        self.branch_out_channels = branch_out_channels
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.refines = nn.ModuleList()
        for _ in self.in_indices:
            self.refines.append(ConvModule(
                self.in_channels,
                self.branch_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=self.norm_cfg))

        mask_branch = []
        for _ in range(self.branch_convs):
            mask_branch.append(ConvModule(
                self.branch_channels,
                self.branch_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=self.norm_cfg))
        mask_branch.append(
            nn.Conv2d(self.branch_channels, self.branch_out_channels, 1))
        self.mask_branch = nn.Sequential(*mask_branch)

    def forward(self, x):
        mask_stride = self.strides[self.in_indices[0]]
        mask_x = self.refines[0](x[self.in_indices[0]])
        for i in range(1, len(self.in_indices)):
            stride = self.strides[i]
            assert stride % mask_stride == 0
            p_x = self.refines[i](x[self.in_indices[i]])
            p_x = aligned_bilinear(p_x, stride // mask_stride)
            mask_x = mask_x + p_x
        return self.mask_branch(mask_x)


@HEADS.register_module()
class CondInstMaskHead(BaseModule):

    def __init__(self,
                 in_channels=8,
                 in_stride=8,
                 out_stride=4,
                 dynamic_convs=3,
                 dynamic_channels=8,
                 disable_rel_coors=False,
                 bbox_head_channels=256,
                 sizes_of_interest=[64, 128, 256, 512, 1024],
                 max_proposals=500,
                 topk_per_img=-1,
                 boxinst_enabled=False,
                 bottom_pixels_removed=10,
                 pairwise_size=3,
                 pairwise_dilation=2,
                 pairwise_color_thresh=0.3,
                 pairwise_warmup=10000,
                 norm_cfg=dict(
                     type='BN',
                     requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer="Conv2d",
                     std=0.01,
                     bias=0)):
        super(CondInstMaskHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        assert in_stride >= out_stride
        assert in_stride % out_stride == 0
        self.in_stride = in_stride
        self.out_stride = out_stride
        assert dynamic_channels > 1
        self.dynamic_convs = dynamic_convs
        self.dynamic_channels = dynamic_channels
        self.disable_rel_coors = disable_rel_coors
        dy_weights, dy_biases = [], []
        dynamic_in_channels = in_channels if disable_rel_coors \
                else in_channels + 2
        for i in range(dynamic_convs):
            in_chn = dynamic_in_channels if i == 0 else \
                    dynamic_channels
            out_chn = 1 if i == dynamic_convs - 1 else \
                    dynamic_channels
            dy_weights.append(in_chn * out_chn)
            dy_biases.append(out_chn)
        self.dy_weights = dy_weights
        self.dy_biases = dy_biases
        self.num_gen_params = sum(dy_weights) + sum(dy_biases)
        self.bbox_head_channels = bbox_head_channels

        self.register_buffer("sizes_of_interest", torch.tensor(sizes_of_interest))
        assert max_proposals == -1 or topk_per_img == -1, \
                'max_proposals and topk_per_img cannot be used at the same time'
        self.max_proposals = max_proposals
        self.topk_per_img = topk_per_img

        self.boxinst_enabled = boxinst_enabled
        self.bottom_pixels_removed = bottom_pixels_removed
        self.pairwise_size = pairwise_size
        self.pairwise_dilation = pairwise_dilation
        self.pairwise_color_thresh = pairwise_color_thresh

        self.register_buffer("_iter", torch.zeros([1]))
        self._warmup_iters = pairwise_warmup

        self.norm_cfg = norm_cfg
        self.fp16_enable = False
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.param_conv = nn.Conv2d(
            self.bbox_head_channels,
            self.num_gen_params,
            3,
            stride=1,
            padding=1)

    def parse_dynamic_params(self, params):
        num_insts = params.size(0)
        params_list = list(torch.split_with_sizes(
            params, self.dy_weights + self.dy_biases, dim=1))
        weights_list = params_list[:self.dynamic_convs]
        biases_list = params_list[self.dynamic_convs:]

        for i in range(self.dynamic_convs):
            if i < self.dynamic_convs - 1:
                weights_list[i] = weights_list[i].reshape(
                    num_insts * self.dynamic_channels, -1, 1, 1)
                biases_list[i] = biases_list[i].reshape(
                    num_insts * self.dynamic_channels)
            else:
                weights_list[i] = weights_list[i].reshape(num_insts * 1, -1, 1, 1)
                biases_list[i] = biases_list[i].reshape(num_insts * 1)
        return weights_list, biases_list

    def forward(self, feat, params, coors, level_inds, img_inds):
        mask_feat = feat[img_inds]
        N, _, H, W = mask_feat.size()
        if not self.disable_rel_coors:
            shift_x = torch.arange(0, W * self.in_stride, step=self.in_stride,
                                   dtype=mask_feat.dtype, device=mask_feat.device)
            shift_y = torch.arange(0, H * self.in_stride, step=self.in_stride,
                                   dtype=mask_feat.dtype, device=mask_feat.device)
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            locations = torch.stack([shift_x, shift_y], dim=0) + self.in_stride // 2

            rel_coors = coors[..., None, None] - locations[None]
            soi = self.sizes_of_interest.float()[level_inds]
            rel_coors = rel_coors / soi[..., None, None, None]
            mask_feat = torch.cat([rel_coors, mask_feat], dim=1)

        weights, biases = self.parse_dynamic_params(params)
        x = mask_feat.reshape(1, -1, H, W)
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=N)
            if i < self.dynamic_convs - 1:
                x = F.relu(x)
        x = x.permute(1, 0, 2, 3)
        x = aligned_bilinear(x, self.in_stride // self.out_stride)
        return x

    def training_sample(self,
                        cls_scores,
                        centernesses,
                        param_preds,
                        coors,
                        level_inds,
                        img_inds,
                        gt_inds):
        num_imgs = param_preds[0].size(0)
        param_preds = torch.cat([
            param_pred.permute(0, 2, 3, 1).flatten(end_dim=2)
            for param_pred in param_preds], dim=0)

        pos_mask = gt_inds != -1
        param_preds = param_preds[pos_mask]
        coors = coors[pos_mask]
        level_inds = level_inds[pos_mask]
        img_inds = img_inds[pos_mask]
        gt_inds = gt_inds[pos_mask]

        if self.max_proposals != -1:
            num_proposals = min(self.max_proposals, param_preds.size(0))
            sampled_inds = torch.randperm(
                num_proposals, device=param_preds.device).long()
        elif self.topk_per_img != -1:
            cls_scores = torch.cat([
                cls_score.permute(0, 2, 3, 1).flatten(end_dim=2)
                for cls_score in cls_scores], dim=0)
            cls_scores = cls_scores[pos_mask]
            centerness = torch.cat([
                centerness.permute(0, 2, 3, 1).reshape(-1)
                for centerness in centernesses], dim=0)
            centerness = centerness[pos_mask]

            sampled_inds = []
            inst_inds = torch.arange(param_preds.size(0), device=param_preds.device)
            for img_id in range(num_imgs):
                img_mask = img_inds == img_id
                if not img_mask.any():
                    continue
                img_gt_inds = gt_inds[img_mask]
                img_inst_inds = inst_inds[img_mask]
                unique_gt_inds = img_gt_inds.unique()
                inst_per_gt = max(int(self.topk_per_img / unique_gt_inds.size(0)), 1)

                for gt_ind in unique_gt_inds:
                    gt_mask = img_gt_inds ==  gt_ind
                    img_gt_inst_inds = img_inst_inds[gt_mask]
                    if img_gt_inst_inds.size(0) > inst_per_gt:
                        cls_scores_ = cls_scores[img_mask][gt_mask]
                        cls_scores_ = cls_scores_.sigmoid().max(dim=1)[0]
                        centerness_ = centerness[img_mask][gt_mask]
                        centerness_ = centerness_.sigmoid()
                        inds = (cls_scores_ * centerness_).topk(inst_per_gt, dim=0)[1]
                        img_gt_inst_inds = img_gt_inst_inds[inds]
                    sampled_inds.append(img_gt_inst_inds)
            sampled_inds = torch.cat(sampled_inds, dim=0)

        param_preds = param_preds[sampled_inds]
        coors = coors[sampled_inds]
        level_inds = level_inds[sampled_inds]
        img_inds = img_inds[sampled_inds]
        gt_inds = gt_inds[sampled_inds]
        return param_preds, coors, level_inds, img_inds, gt_inds






    @force_fp32(apply_to=('mask_logits', )) # TODO add the sem_loss
    def loss(self,
             imgs,
             img_metas,
             mask_logits,
             gt_inds,
             gt_bboxes,
             gt_masks,
             gt_labels):
        self._iter += 1
        similarities, gt_bitmasks, bitmasks_full = self.get_targets(gt_bboxes, imgs, img_metas)
        mask_scores = mask_logits.sigmoid()
        gt_bitmasks = torch.cat(gt_bitmasks, dim=0)
        gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(1).to(mask_scores.dtype)

        losses= {}

        if len(mask_scores) == 0: #there is no instances detected
            dummy_loss = 0 * mask_scores.sum()
            if not self.boxinst_enabled:
                losses["loss_mask"] = dummy_loss
            else:
                losses["loss_prj"] = dummy_loss
                losses["loss_pairwise"] = dummy_loss

        if self.boxinst_enabled:
            img_color_similarity = torch.cat(similarities, dim=0)
            img_color_similarity = img_color_similarity[gt_inds].to(dtype=mask_scores.dtype)

            loss_prj_term = compute_project_term(mask_scores, gt_bitmasks)
            
            pairwise_losses = compute_pairwise_term(mask_logits, self.pairwise_size, self.pairwise_dilation)

            weights = (img_color_similarity >= self.pairwise_color_thresh).float() * gt_bitmasks.float()

            loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)

            warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)
            loss_pairwise = loss_pairwise * warmup_factor

            losses.update({
                "loss_prj": loss_prj_term,
                "loss_pairwise": loss_pairwise,
            })

        return losses




    def get_targets(self, gt_bboxes, img, img_metas):
        """get targets"""

        if self.boxinst_enabled:

            padded_image_masks = []
            padded_images = []

            for i in range(len(img_metas)):
                original_image_masks = torch.ones(img_metas[i]['img_shape'][:2], dtype=torch.float32, device=img.device)

                im_h = img_metas[i]['ori_shape'][0]
                pixels_removed = int(
                    self.bottom_pixels_removed * float(img_metas[i]['img_shape'][0]) / float(im_h)
                )
                if pixels_removed > 0:
                    original_image_masks[-pixels_removed:, :] = 0

                padding = (0, img.shape[3] - img_metas[i]['img_shape'][1],
                           0, img.shape[2] - img_metas[i]['img_shape'][0])  #[left, right, top, bottom]

                padded_image_mask = F.pad(original_image_masks, pad=padding)
                padded_image_masks.append(padded_image_mask)

                original_image = get_original_image(img[i], img_metas[i])  # get RGB image tensor
                # original_image = original_image.to(img.device)
                padded_image = F.pad(original_image, pad=padding)
                padded_images.append(padded_image)

            padded_image_masks = torch.stack(padded_image_masks, dim=0)
            padded_images = torch.stack(padded_images, dim=0)

            similarities, bitmasks, bitmasks_full = self.get_bitmasks_from_boxes(gt_bboxes, padded_images, padded_image_masks)

            return similarities, bitmasks, bitmasks_full

        else:
            pass #TODO


    def get_bitmasks_from_boxes(self, gt_bboxes, padded_images, padded_image_masks):
        h, w = padded_images.shape[2:]
        stride = self.out_stride
        start = int(stride // 2)

        assert padded_images.size(2) % stride == 0
        assert padded_images.size(3) % stride == 0

        downsampled_images = F.avg_pool2d(padded_images.float(), kernel_size=stride, stride=stride, padding=0)
        downsampled_image_masks = padded_image_masks[:, start::stride, start::stride]

        similarities = []
        bitmasks = []
        bitmasks_full = []

        for i, per_img_gt_bboxes in enumerate(gt_bboxes):
            image_lab = color.rgb2lab(downsampled_images[i].byte().permute(1, 2, 0).cpu().numpy())
            image_lab = torch.as_tensor(image_lab, device=padded_image_masks.device, dtype=torch.float32)
            image_lab = image_lab.permute(2, 0, 1)[None]
            image_color_similarity = get_image_color_similarity(
                image_lab, downsampled_image_masks[i],
                self.pairwise_size, self.pairwise_dilation
            )

            per_im_bitmasks = []
            per_im_bitmasks_full = []

            for per_box in per_img_gt_bboxes:  # [x1,y1, x2, y2]
                bitmask_full = torch.zeros((h, w), device=per_box.device).float()
                bitmask_full[int(per_box[1]): int(per_box[3]) + 1, int(per_box[0]):int(per_box[2]) + 1] = 1.0

                bitmask = bitmask_full[start::stride, start::stride]

                assert bitmask.size(0) * stride == h
                assert bitmask.size(1) * stride == w

                per_im_bitmasks.append(bitmask)
                per_im_bitmasks_full.append(bitmask_full)

            per_im_bitmasks = torch.stack(per_im_bitmasks, dim=0)
            per_im_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)

            similarities.append(torch.cat([image_color_similarity for _ in range(len(per_img_gt_bboxes))], dim=0))
            bitmasks.append(per_im_bitmasks)
            bitmasks_full.append(per_im_bitmasks_full)

        return similarities, bitmasks, bitmasks_full
