import cv2


def cv_show(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


# 读取两个图像
template = cv2.imread('muban5.jpg', 0)  # 读取灰度图
cv_show('img', template)  # 展示图象
img = cv2.imread('image2.jpg', 0)
cv_show('img', img)

# 获取小图像的高和宽
h, w = template.shape[:10]

# 不同的方法模板匹配的方式不同
methodology = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF,
               cv2.TM_SQDIFF_NORMED]
# cv2.TM_CCOEFF：相关系数匹配。该方法计算输入图像和模板图像之间的相关系数，值越大表示匹配程度越好。
# cv2.TM_CCOEFF_NORMED：标准归一化相关系数匹配。该方法计算输入图像和模板图像之间的标准归一化相关系数，也就是相关系数除以两个图像各自的标准差的乘积。值越大表示匹配程度越好。
# cv2.TM_CCORR：相关性匹配。该方法计算输入图像和模板图像之间的相关性，值越大表示匹配程度越好。
# cv2.TM_CCORR_NORMED：标准归一化相关性匹配。该方法计算输入图像和模板图像之间的标准归一化相关性，也就是相关性除以两个图像各自的标准差的乘积。值越大表示匹配程度越好。
# cv2.TM_SQDIFF：平方差匹配。该方法计算输入图像和模板图像之间的平方差，值越小表示匹配程度越好。
# cv2.TM_SQDIFF_NORMED：标准归一化平方差匹配。该方法计算输入图像和模板图像之间的标准归一化平方差，也就是平方差除以两个图像各自的标准差的乘积。值越小表示匹配程度越好。

method = methodology[1]  # 选了一个cv2.TM_CCOEFF_NORMED方法进行图像匹配，匹配方式为比较模板和图像中各个区域的标准归一化相关系数
res = cv2.matchTemplate(img, template, method)  # 比较完成的结果存储在res中，是一个ndarray类型的

# 获取匹配结果中的最大值和最小值
# 通过左上角点的位置，加上之前获取的小图像的宽和高h,w，就可以把图像在原图中的那个位置框出来了
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # 最主要的是知道图像位置_loc

# 不同的匹配算法，计算匹配到的位置的方式不同
if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    top_left = min_loc
else:
    top_left = max_loc

# 通过左上角点的坐标和h，w计算右下角点的坐标
bottom_right = (top_left[0] + w, top_left[1] + h // 2)  # [0]为横坐标，[1]为纵坐标，横加宽纵加高就是右下角的点坐标

# 绘制矩形
resoult_img = cv2.rectangle(img.copy(), top_left, bottom_right, 255, 1)

cv_show('img', resoult_img)