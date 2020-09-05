# 由于阿里天池上传图片麻烦，所以预先用ImageProcessor.py将图片处理好之后存进csv再进行上传训练。

# ResNet + MultiLabelSoftMarginLoss

## 使用MultiLabelSoftMarginLoss进行多标签多分类，需要对label进行onehot编码。ImageCaptcha生成的验证码图片比较复杂，40000张图片最终的准确率为80%，结果还是令人满意的。若增大训练集，准确率应该还能显著提高。