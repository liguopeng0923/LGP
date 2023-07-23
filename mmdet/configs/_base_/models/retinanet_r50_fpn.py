# model settings
# https://zhuanlan.zhihu.com/p/346198300
model = dict(
    type='RetinaNet',
    backbone=dict(
        # 骨架网络名
        type='ResNet',
        # 使用ResNet50
        depth=50,
        # 包括 stem+ 4个 stage 输出
        num_stages=4,
        # 表示本模块输出的特征图索引，（0，1，2，3）表示4个stage输出都需要
        # stride为（4，8，16，32），channel 为（256，512，1024，2048）
        #ResNet 提出了骨架网络设计范式即 stem+n stage+ cls head，对于 ResNet 而言，其实际 forward 流程是 stem -> 4 个 stage -> 分类 head，stem 的输出 stride 是 4，而 4 个 stage 的输出 stride 是 4,8,16,32，这 4 个输出就对应 out_indices 索引。例如如果你想要输出 stride=4 的特征图，那么你可以设置 out_indices=(0,)，如果你想要输出 stride=4 和 8 的特征图，那么你可以设置 out_indices=(0, 1)。
        out_indices=(0, 1, 2, 3),
        # 表示固定 stem 加上第一个stage的权重，不进行训练
        # 该参数表示你想冻结前几个 stages 的权重，ResNet 结构包括 stem+4 stage
        # frozen_stages=-1，表示全部可学习
        # frozen_stage=0，表示stem权重固定
        # frozen_stages=1，表示 stem 和第一个 stage 权重固定
        # frozen_stages=2，表示 stem 和前两个 stage 权重固定
        # 依次类推
        frozen_stages=1,
        # 所有的 BN 层的可学习参数都需要梯度
        # norm_cfg 表示所采用的归一化算子，一般是 BN 或者 GN，而 requires_grad 表示该算子是否需要梯度，也就是是否进行参数更新，而布尔参数 norm_eval 是用于控制整个骨架网络的归一化算子是否需要变成 eval 模式。
        norm_cfg=dict(type='BN', requires_grad=True),
        # backbone 所有的 BN 层的均值和方差都直接采用全局预训练值，不进行更新
        norm_eval=True,
        # 默认采用 pytorch模式
        style='pytorch',
        # 使用 pytorch 提供的在 imagenet 上面训练过的权重作为预训练权重
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        # 前面说过 ResNet 输出 4 个不同尺度特征图 (c2,c3,c4,c5)，stride 分别是 (4,8,16,32)，通道数为 (256,512,1024,2048),通过配置文件我们可以知道：

        # start_level=1 说明虽然输入是 4 个特征图，但是实际上 FPN 中仅仅用了后面三个
        # num_outs=5 说明 FPN 模块虽然是接收 3 个特征图，但是输出 5 个特征图
        # add_extra_convs='on_input' 说明额外输出的 2 个特征图的来源是骨架网络输出，而不是 FPN 层本身输出又作为后面层的输入
        # out_channels=256 说明了 5 个输出特征图的通道数都是 256
        
        # 下面对代码运行流程进行描述：
        # 将 c3、c4 和 c5 三个特征图全部经过各自 1x1 卷积进行通道变换得到 m3~m5，输出通道统一为 256
        # 从 m5(特征图最小)开始，先进行 2 倍最近邻上采样，然后和 m4 进行 add 操作，得到新的 m4
        # 将新 m4 进行 2 倍最近邻上采样，然后和 m3 进行 add 操作，得到新的 m3
        # 对 m5 和新融合后的 m4、m3，都进行各自的 3x3 卷积，得到 3 个尺度的最终输出 P5～P3
        # 将 c5 进行 3x3 且 stride=2 的卷积操作，得到 P6
        # 将 P6 再一次进行 3x3 且 stride=2 的卷积操作，得到 P7
        # P6 和 P7 目的是提供一个大感受野强语义的特征图，有利于大物体和超大物体检测。 在 RetinaNet 的 FPN 模块中只包括卷积，不包括 BN 和 ReLU。

        # 总结：FPN 模块接收 c3, c4, c5 三个特征图，输出 P3-P7 五个特征图，通道数都是 256, stride 为 (8,16,32,64,128)，其中大 stride (特征图小)用于检测大物体，小 stride (特征图大)用于检测小物体。
        type='FPN',
        # ResNet 模块输出的4个尺度特征图通道数
        in_channels=[256, 512, 1024, 2048],
        # FPN 输出的每个尺度输出特征图通道
        out_channels=256,
        # 从输入多尺度特征图的第几个开始计算
        start_level=1,
        # 额外输出层的特征图来源
        add_extra_convs='on_input',
        # FPN 输出特征图个数
        num_outs=5),
    # 输出头包括分类和检测两个分支，且每个分支都包括 4 个卷积层，不进行参数共享，分类 Head 输出通道是 num_class*K，检测 head 输出通道是4*K, K 是 anchor 个数, 虽然每个 Head 的分类和回归分支权重不共享，但是 5 个输出特征图的 Head 模块权重是共享的。
    bbox_head=dict(
        type='RetinaHead',
        # COCO数据集类别数
        num_classes=80,
        # FPN 层输出特征图通道数
        in_channels=256,
        # 每个分支堆叠四层卷积
        stacked_convs=4,
        # 中间特征图通道数
        feat_channels=256,
        # anchor生成
        # 遍历 m 个输出特征图，在每个特征图的 (0,0) 或者说原图的 (0,0) 坐标位置生成 base_anchors，注意 base_anchors 不是特征图尺度，而是原图尺度
        # 遍历 m 个输出特征图中每个特征图上每个坐标点，将其映射到原图坐标上原图坐标点加上 base_anchors，就可以得到特征图每个位置的对应到原图尺度的 anchor 列表，anchor 列表长度为 m
        anchor_generator=dict(
            type='AnchorGenerator',
            # 特征图 anchor 的base scale，值越大，所有anchor的尺度都会变大
            octave_base_scale=4,
            # 每个特征图有三个尺度，2**0，2**（1/3），2**（2/3）
            scales_per_octave=3,
            # 每个特征图有三个高宽比例
            ratios=[0.5, 1.0, 2.0],
            # 特征图对应的 stride 即特征图相对于原图下采样的比例 ， 必须特征图 stride 一致，不可随意更改
            strides=[8, 16, 32, 64, 128]),
        # 在 anchor-based 算法中，为了利用 anchor 信息进行更快更好的收敛，一般会对 head 输出的 bbox 分支 4 个值进行编解码操作，作用有两个：

        # 更好的平衡分类和回归分支 loss，以及平衡 bbox 四个预测值的 loss
        # 训练过程中引入 anchor 信息，加快收敛
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        # (1) 初始化所有 anchor 为忽略样本，假设所有输出特征的所有 anchor 总数一共 n 个，对应某张图片中 gt bbox 个数为 m，首先初始化长度为 n 的 assigned_gt_inds，全部赋值为 -1，表示当前全部设置为忽略样本
        # (2) 计算背景样本，将每个 anchor 和所有 gt bbox 计算 iou，找出最大 iou，如果该 iou 小于 neg_iou_thr 或者在背景样本阈值范围内，则该 anchor 对应索引位置的 assigned_gt_inds 设置为 0，表示是负样本(背景样本)
        # (3) 计算高质量正样本，将每个 anchor 和所有 gt bbox 计算 iou，找出最大 iou，如果其最大 iou 大于等于 pos_iou_thr，则设置该 anchor 对应所有的 assigned_gt_inds 设置为当前匹配 gt bbox 的编号 +1(后面会减掉 1)，表示该 anchor 负责预测该 gt bbox，且是高质量 anchor。之所以要加 1，是为了区分背景样本(背景样本的 assigned_gt_inds 值为 0)
        # (4) 适当增加更多正样本，在第三步计算高质量正样本中可能会出现某些 gt bbox 没有分配给任何一个 anchor (由于 iou 低于 pos_iou_thr)，导致该 gt bbox 不被认为是前景物体，此时可以通过 self.match_low_quality=True 配置进行补充正样本。对于每个 gt bbox 需要找出和其最大 iou 的 anchor 索引，如果其 iou 大于 min_pos_iou，则将该 anchor 对应索引的 assigned_gt_inds 设置为正样本，表示该 anchor 负责预测对应的 gt bbox。通过本步骤，可以最大程度保证每个 gt bbox 都有相应的 anchor 负责预测，但是如果其最大 iou 值还是小于 min_pos_iou，则依然不被认为是前景物体。
        # 从这一步可以看出，3 和 4 有部分 anchor 重复分配了，即当某个 gt bbox 和 anchor 的最大 iou 大于等于 pos_iou_thr，那肯定大于 min_pos_iou，此时 3 和 4 步骤分配的同一个 anchor，并且从上面注释可以看出本步骤可能会引入低质量 anchor，是否需要开启本步骤需要根据不同算法来确定。
        # 此时可以可以得到如下总结：
        # 如果 anchor 和所有 gt bbox 的最大 iou 值小于 0.4，那么该 anchor 就是背景样本
        # 如果 anchor 和所有 gt bbox 的最大 iou 值大于等于 0.5，那么该 anchor 就是高质量正样本
        # 如果 gt bbox 和所有 anchor 的最大 iou 值大于等于 0(可以看出每个 gt bbox 都一定有至少一个 anchor 匹配)，那么该 gt bbox 所对应的 anchor 也是正样本
        # 其余样本全部为忽略样本即 anchor 和所有 gt bbox 的最大 iou 值处于 [0.4,0.5) 区间的 anchor 为忽略样本，不计算 loss
        assigner=dict(
            # 最大 IOU 原则分配器
            type='MaxIoUAssigner',
            # 正样本阈值
            pos_iou_thr=0.5,
            # 负样本阈值
            neg_iou_thr=0.4,
            # 正样本阈值下限
            min_pos_iou=0,
            # 忽略 bboes 的阈值，-1表示不忽略
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    # (1) 对 5 个 head 输出特征图结果进行遍历，先按照预测分值排序，保留前 nms_pre 个预测结果
    # (2) 对剩下的 bbox 进行解码
    # (3) 还原到原图尺度
    # (4) 用 score_thr 阈值对所有结果进行过滤，然后将保留框进行 nms，最终输出框最大为 max_per_img 个

    test_cfg = dict(
        # nms 前每个输出层最多保留1000个预测框
        nms_pre=1000,
        # 过滤掉的最小 bbox 尺寸
        min_bbox_size=0,
        # 分值阈值
        score_thr=0.05,
        # nms 方法和 nms 阈值
        nms=dict(type='nms', iou_threshold=0.5),
        # 最终输出的每张图片最多 bbox 个数
        max_per_img=100))
