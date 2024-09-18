from PIL import Image

# 載入圖片
image_path = "in2.png"  # 替換為你的圖片路徑
img = Image.open(image_path)

# 獲取圖片的寬高
img_width, img_height = img.size

# 設定行列數量
cols, rows = 10, 6  # 10 列，6 行

# 計算每張圖片的寬高
single_img_width = img_width // cols
single_img_height = img_height // rows

# 迴圈分割並儲存每張圖片
for row in range(rows):
    for col in range(cols):
        # 計算每個圖片的左上角和右下角的坐標
        left = col * single_img_width
        upper = row * single_img_height
        right = left + single_img_width
        lower = upper + single_img_height

        # 裁剪出每張圖片
        cropped_img = img.crop((left, upper, right, lower))

        # 儲存每張圖片
        output_path = f"../test/2_{row}_{col}.png"
        cropped_img.save(output_path)
        print(f"已儲存 {output_path}")

print("圖片分割完成！")
