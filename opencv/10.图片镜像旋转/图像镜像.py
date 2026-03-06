import cv2

if __name__ == "__main__":
    path = "./lena.png"
    image_np = cv2.imread(path)
    # 对图片进行镜像操作
    mirroring_image = cv2.flip(image_np, 0)
    cv2.imshow("mirroring_image", mirroring_image)
    cv2.waitKey(0)
