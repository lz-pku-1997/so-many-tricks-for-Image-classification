# so-many-tricks-for-Image-classification
so many tricks in tf.keras

使用时只需针对自己的数据更改数据接口就行

# 更新历史
# v1.0  
baseline efficientnet b0

including: random_flip_left_right; random_flip_up_down; random_brightness; random_contrast; random_hue; random_saturation;

cutmix; mixup; gridmask; pre_trained; dense_activation; lr_scheduler; tta; focal_loss; label_smoothing;

# v1.1
交叉验证、增加映射到等差数列的融合规则（该融合方式使植物病理学auc由0.978提升到了0.982,而正常求和平均不能提升）




# 程序默认数据集地址：
狗品种分类：  https://www.kaggle.com/c/dog-breed-identification/overview

植物病理学分类： https://www.kaggle.com/c/plant-pathology-2020-fgvc7/overview
