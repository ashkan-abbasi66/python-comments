





"""
See
https://github.com/weiliu89/VOCdevkit/blob/master/example_segmenter.m
https://pypi.org/project/pascal-voc/

for evaluation on PASCAL VOC dataset.
-------------------------------------------

A solution to obtain IoU values for
Labels are determined by three RGB values
Convert them to scalar outputs.

For example,
[0, 0, 0]     ===> 0
[128, 0, 0]   ===> 1
[0, 128, 0]   ===> 2
...

Then, use
    for k in num_classes:
        label = label(label==k)
        pred = pred(pred ==k)
        iou_val = iou_coef(label_image.unsqueeze(0), pred.cpu().permute(2, 0, 1).unsqueeze(0))
    compute mean of iou_val.
---------------------
    # iou_val = iou_coef(label_image.unsqueeze(0), pred.cpu().permute(2, 0, 1).unsqueeze(0))
    # print("iou:", iou_val)
    # iou_list.append(iou_val)

# print("mean iou = ", sum(iou_list)/len(iou_list))
"""
