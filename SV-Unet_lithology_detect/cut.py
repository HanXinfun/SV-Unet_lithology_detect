from osgeo import gdal

# 打开原始TIFF图像
input_file = "E:/IMAGEP/out3.tif"
dataset = gdal.Open(input_file, gdal.GA_ReadOnly)

print("1")
if dataset is None:
    print("无法打开输入文件")
else:
    # 获取原始图像的宽度和高度
    num_bands = dataset.RasterCount
    print(f"图像波段数: {num_bands}")
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    # 定义裁剪尺寸
    crop_width = 395
    crop_height = 395

    # 计算裁剪数量
    num_crops_width = width // crop_width
    num_crops_height = height // crop_height

    # 设置裁剪后的图像保存选项
    options = [
        'TILED=YES',
        'COMPRESS=LZW',
        #'PHOTOMETRIC=RGB',
        'PROFILE=GeoTIFF',
    ]

    for i in range(num_crops_width):
        for j in range(num_crops_height):
            left = i * crop_width
            upper = j * crop_height
            right = left + crop_width
            lower = upper + crop_height

            # 创建输出图像
            output_file = f"E:/IMAGEP/tif/output_{j+1}_{i+1}.tif"
            output_dataset = gdal.GetDriverByName('GTiff').Create(
                output_file,
                crop_width,
                crop_height,
                1,  # 3表示RGB通道
                #gdal.GDT_Byte,
                dataset.GetRasterBand(1).DataType,
                options
            )

            # # 设置输出图像的地理变换信息（空间坐标和投影信息）
            output_dataset.SetGeoTransform((
                dataset.GetGeoTransform()[0] + left * dataset.GetGeoTransform()[1],
                dataset.GetGeoTransform()[1],
                0,
                dataset.GetGeoTransform()[3] + upper * dataset.GetGeoTransform()[5],
                0,
                dataset.GetGeoTransform()[5]
            ))

            # 将原始图像的RGB数据写入输出图像
            # for band_index in range(1, 4):  # 1-based index
            #     band = dataset.GetRasterBand(band_index)
            #     output_band = output_dataset.GetRasterBand(band_index)
            #     data = band.ReadAsArray(left, upper, crop_width, crop_height)
            #     output_band.WriteArray(data)
            band = dataset.GetRasterBand(1)
            output_band = output_dataset.GetRasterBand(1)
            data = band.ReadAsArray(left, upper, crop_width, crop_height)
            output_band.WriteArray(data)

            # 关闭输出图像
            output_dataset = None

    # 关闭原始图像
    dataset = None