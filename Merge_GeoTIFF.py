import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from osgeo import gdal
from scipy.ndimage import gaussian_filter
from PIL import Image
import json
import copy

def main(input_dir, output_dir, sigma, output_graphs, data_excluded, data_excluded_u, data_excluded_l, ue_landscape, small_units, not_flat_earth):

    Image.MAX_IMAGE_PIXELS = None

    # 画像を指定した解像度で分割し、元の画像の位置を中心に配置し、余ったピクセルを黒で塗りつぶす関数
    def split_and_fill_image(array, normalized_array, resolution, small_units, transform):
        # ディープコピーを作成
        array = copy.deepcopy(array)
        normalized_array = copy.deepcopy(normalized_array)

        height, width = array.shape
        x_tiles = width // resolution + 1
        y_tiles = height // resolution + 1

        # 余白のサイズを計算
        padding_x = (resolution * x_tiles - width) // 2
        padding_y = (resolution * y_tiles - height) // 2

        # 奇数の場合、左端と上に1px多くなるように調整
        padding_x += (resolution * x_tiles - width) % 2
        padding_y += (resolution * y_tiles - height) % 2

        # 全体の配列を生成
        full_array = np.pad(array, ((padding_y, padding_y), (padding_x, padding_x)), 'constant', constant_values=0)
        full_normalized_array = np.pad(normalized_array, ((padding_y, padding_y), (padding_x, padding_x)), 'constant', constant_values=0)

        # タイルの境界を一致させる
        if not small_units:
            for j in range(y_tiles):
                for i in range(x_tiles):
                    left = i * resolution
                    upper = j * resolution
                    right = left + resolution
                    lower = upper + resolution
                    if i != x_tiles - 1:
                        for k in range(upper, lower):
                            full_array[k, right-1:right+1] = np.mean(full_array[k, right-2:right+2])
                            full_normalized_array[k, right-1:right+1] = np.mean(full_normalized_array[k, right-2:right+2])
                    if j != y_tiles - 1:
                        for l in range(left, right):
                            full_array[lower-1:lower+1, l] = np.mean(full_array[lower-2:lower+2, l])
                            full_normalized_array[lower-1:lower+1, l] = np.mean(full_normalized_array[lower-2:lower+2, l])

        tiles = []
        for j in range(y_tiles):
            for i in range(x_tiles):
                left = i * resolution
                upper = j * resolution
                right = left + resolution
                lower = upper + resolution
                new_array = full_array[upper:lower, left:right]
                new_normalized_array = full_normalized_array[upper:lower, left:right]

                # タイルの上端と下端の北緯を計算（パディングを考慮）
                upper_lat = transform[3] - (upper + padding_y) * transform[5]
                lower_lat = transform[3] - (lower + padding_y) * transform[5]

                # Convert the normalized array to an image
                if small_units:
                    valid_mask = new_array != 0

                    #ガウスぼかし
                    if np.any(valid_mask):
                        new_array[valid_mask] = gaussian_filter(new_array[valid_mask], sigma=sigma)

                    # Normalize the array
                    if np.any(valid_mask):
                        min_val = new_array[valid_mask].min()
                        max_val = new_array[valid_mask].max()
                        if max_val != min_val:
                            with np.errstate(divide='ignore'):
                                normalized_array = np.where(valid_mask, (new_array - min_val) / (max_val - min_val), 0)
                        else:
                            normalized_array = np.zeros_like(new_array)
                    else:
                        normalized_array = np.zeros_like(new_array)
                    
                    new_img = Image.fromarray((normalized_array * 65535).astype(np.uint16))
                else:
                    # Use the pre-computed normalized array
                    normalized_array = new_normalized_array

                    new_img = Image.fromarray((normalized_array).astype(np.uint16))

                tiles.append((new_img, new_array, upper_lat, lower_lat))
        return tiles, x_tiles, y_tiles
    
    print('Loding GeoTIFF...')

    # 入力ファイルのリストを作成
    file_list = glob.glob(os.path.join(input_dir, '*.tif'))

    # GeoTIFFファイルが存在するかどうかを確認
    if not file_list:
        raise FileNotFoundError(f'No GeoTIFF files found in the directory: {input_dir}')

    # gdal_merge.pyのオプションを設定
    merge_options = gdal.BuildVRTOptions(resampleAlg='nearest')

    print("Merging now...")

    # GeoTIFFファイルを結合
    merged = gdal.BuildVRT(os.path.join(output_dir, 'output.vrt'), file_list, options=merge_options)
    gdal.Translate(os.path.join(output_dir, 'output.tif'), merged)

    # 結合したGeoTIFFファイルを読み込む
    ds = gdal.Open(os.path.join(output_dir, 'output.tif'))

    # 変換パラメータを取得
    transform_D = ds.GetGeoTransform()

    # バンドの数を確認
    num_bands = ds.RasterCount

    # バンドの数に基づいてデータを読み込む
    if num_bands >= 1:
        band = ds.GetRasterBand(1)
        array = band.ReadAsArray()
        unit_type = band.GetUnitType()
        if unit_type == '':
            unit_type = 'meter'
    else:
        raise ValueError('The GeoTIFF file does not contain any bands.')


    if output_graphs:
        # ヒストグラムと箱ひげ図を作成
        plt.figure(figsize=(12, 6))

        plt.subplot(121)
        plt.hist(array.flatten(), bins=100, color='blue', alpha=0.7)
        plt.title('Histogram of Elevation')

        plt.subplot(122)
        sns.boxplot(y=array.flatten())
        plt.title('Boxplot of Elevation')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'elevation_distribution.png'))
        plt.close()

    # 上位と下位の0.5%の閾値を計算
    lower_threshold = np.percentile(array, float((data_excluded + data_excluded_l)))
    upper_threshold = np.percentile(array, float(100-(data_excluded + data_excluded_u)))

    # 閾値を超える値を閾値に置き換える
    array = np.where(array < lower_threshold, lower_threshold, array)
    array = np.where(array > upper_threshold, upper_threshold, array)

    # ガウスぼかしを適用
    array = gaussian_filter(array, sigma=sigma)

    # 地球のように丸くする
    if not_flat_earth:
        # 地球の半径 (m or km)
        R_equator = 6378.1370  # km
        R_pole = 6356.7523  # km

        # 単位タイプに基づいて地球の半径を調整
        if unit_type == 'meter':
            R_equator *= 1000  # km to m
            R_pole *= 1000  # km to m

        # 緯度の配列を作成
        height, width = array.shape
        latitudes = np.linspace(transform_D[3], transform_D[3] + height*transform_D[5], height)

        # 緯度をラジアンに変換
        latitudes_rad = np.deg2rad(latitudes)

        # 各緯度での地球の半径を計算
        R = np.sqrt(((R_equator**2 * np.cos(latitudes_rad))**2 + (R_pole**2 * np.sin(latitudes_rad))**2) / 
                    ((R_equator * np.cos(latitudes_rad))**2 + (R_pole * np.sin(latitudes_rad))**2))

        # 高度データに地球の半径からの相対的な長さを加える
        array += (R - R_pole)

        

    # 正規化した標高データを基に色分けしたPNG画像を出力
    normalized_array = (((array - array.min()) / (array.max() - array.min())* 65535).astype(np.uint16))
    plt.imsave(os.path.join(output_dir, 'output.png'), (normalized_array), cmap='gray')

    print(f'GeoTIFF files have been merged into: {os.path.join(output_dir, "output.tif")}')
    print('A PNG image has been created based on the elevation data.')

    # UE4のランドスケープ用のスケールを計算して出力
    if ue_landscape:
        # 最も高い点と最も低い点の差を求める
        elevation_diff = array.max() - array.min()

        # 差をcm単位に変換し、1/512する
        z_scale = elevation_diff * 100 / 512

        # 結果をテキストファイルに出力
        with open(os.path.join(output_dir, 'output.txt'), 'w', encoding='utf-8') as f:
            f.write(f'x resolution = {abs(ds.GetGeoTransform()[1])} M/px\ny resolution ={abs(ds.GetGeoTransform()[5])} M/px\nmax height = {array.max()} M\nmin height = {array.min()} M\nUE z scale = {z_scale}')

        print(f'A text file with the basic data created: {os.path.join(output_dir, "output.txt")}')

        # 解像度ごとに画像を分割して出力
        resolutions = [127, 253, 505, 1009, 2017, 4033, 8129]
        for res in resolutions:
            # 分割後の画像を保存するディレクトリを作成
            res_dir = os.path.join(output_dir, f'resolution_{res}x{res}')
            os.makedirs(res_dir, exist_ok=True)

            # 画像を指定した解像度で分割し、余白部分を黒で塗りつぶす
            tiles, x_tiles, y_tiles = split_and_fill_image(array, normalized_array, res, small_units, transform_D)

            # 分割した画像を保存
            data = {}
            data['sprit_res']=res
            data['map_x_res']=abs(ds.GetGeoTransform()[1])
            data['map_y_res']=abs(ds.GetGeoTransform()[5])
            data['x_tiles']=x_tiles
            data['y_tiles']=y_tiles

            for i, (tile, tile_array, upper_lat, lower_lat) in enumerate(tiles):
                tile.save(os.path.join(res_dir, f'{i}.png'))

                if small_units:

                    # Calculate the difference between the highest and lowest points
                    valid_mask = tile_array != 0
                    elevation_diff = tile_array[valid_mask].max() - tile_array[valid_mask].min() if np.any(valid_mask) else 0

                    # Convert the difference to cm and divide by 512
                    z_scale = elevation_diff * 100 / 512

                    # Store the result in a dictionary
                    data[i] = {
                        'z_scale': z_scale,
                        'upper_lat': float(upper_lat),
                        'lower_lat': float(lower_lat),
                        'max_height': float(tile_array[valid_mask].max() if np.any(valid_mask) else 0),
                        'min_height': float(tile_array[valid_mask].min() if np.any(valid_mask) else 0),
                    }
                else:
                    data[i] = {
                        'z_scale': z_scale,
                        'upper_lat': float(upper_lat),
                        'lower_lat': float(lower_lat),
                        'max_height': float(array.max()),
                        'min_height': float(array.min()),
                    }


            with open(os.path.join(res_dir, '.data.json'), 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

            print(f'Split maps and data files have been created in: {res_dir}')

if __name__ == '__main__':
    # コマンドラインオプションの設定
    parser = argparse.ArgumentParser(description='Merge GeoTIFF files and create a PNG image.')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing the input GeoTIFF files.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the output files.')
    parser.add_argument('--sigma', type=float, default=0, help='Sigma value for the Gaussian blur. Default is 0 (no blur).')
    parser.add_argument('--output-graphs', action='store_true', help='Output a graph to visually recognise outliers. Can be very memory intensive.')
    parser.add_argument('--data-excluded', type=float, default=0, help='Remove outliers from the data by removing the upper %% and lower %%. Default is 0 (calculated based on all data).')
    parser.add_argument('--data-excluded_u', type=float, default=0, help='NEVER USE WITH --data-excluded. Remove outliers from the data by removing the upper %%. Default is 0.')
    parser.add_argument('--data-excluded_l', type=float, default=0, help='NEVER USE WITH --data-excluded. Remove outliers from the data by removing the lower %%. Default is 0.')
    parser.add_argument('--not-flat-earth', action='store_true', help='Produce height map data along the spherical shape of the earth.')
    parser.add_argument('--ue-landscape', action='store_true', help='Prepare data for Unreal Engine landscapes.')
    parser.add_argument('--small-units', action='store_true', help='ALWAYS USE WITH --ue-landscape. When binarising elevation data, binarise with elevation data for individual tiles.')
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.sigma, args.output_graphs, args.data_excluded, args.data_excluded_u, args.data_excluded_l, args.ue_landscape, args.small_units, args.not_flat_earth)