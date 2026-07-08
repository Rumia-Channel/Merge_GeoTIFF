import os
import glob

from .landscape import split_and_fill_image

DEFAULT_LANDSCAPE_RESOLUTIONS = [127, 253, 505, 1009, 2017, 4033, 8129]


def main(input_dir, output_dir, sigma, output_graphs, data_excluded, data_excluded_u, data_excluded_l, not_flat_earth, ue_landscape, small_units, landscape_res):
    import json

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from osgeo import gdal, osr
    from PIL import Image
    from pyproj import Transformer
    from scipy.ndimage import gaussian_filter
    from tqdm import tqdm

    osr.UseExceptions()
    Image.MAX_IMAGE_PIXELS = None
    os.makedirs(output_dir, exist_ok=True)
    
    print('Loding GeoTIFF...')

    # 入力ファイルのリストを作成
    file_list = glob.glob(os.path.join(input_dir, '*.tif'))

    # GeoTIFFファイルが存在するかどうかを確認
    if not file_list:
        raise FileNotFoundError(f'No GeoTIFF files found in the directory: {input_dir}')

    # gdal_merge.pyのオプションを設定
    merge_options = gdal.BuildVRTOptions(resampleAlg='cubic')

    print("Done")
    print("Merging now...")

    # GeoTIFFファイルを結合
    merged = gdal.BuildVRT(os.path.join(output_dir, 'output.vrt'), file_list, options=merge_options)
    gdal.Translate(os.path.join(output_dir, 'output.tif'), merged)

    # 結合したGeoTIFFファイルを読み込む
    ds = gdal.Open(os.path.join(output_dir, 'output.tif'))

    # 元の座標参照システムのEPSGコードを取得
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(ds.GetProjection())
    src_epsg = src_srs.GetAttrValue("AUTHORITY", 1)

    # 変換器を作成（元のEPSGコードからEPSG:4326へ）
    transforme = Transformer.from_crs(int(src_epsg), 4326, always_xy=True)

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

        # WGS84の定数
        R_equator = 6378137.0  # 赤道半径（m）
        if not unit_type == "meter":
            R_equator = R_equator / 1000
        f = 1 / 298.257223563  # 扁平率
        #R_pole = R_equator * (1 - f)  # 極半径（m）

        # ピクセルのサイズを取得
        cols = ds.RasterXSize
        rows = ds.RasterYSize

        # 緯度と経度の配列を初期化
        lats = np.zeros((rows, cols))
        lons = np.zeros((rows, cols))

        pbar_ac = tqdm(total=rows*cols, desc="Converting to angles... ")
        # 各ピクセルの緯度と経度を計算
        for i in range(rows):
            for j in range(cols):
                lons[i, j], lats[i, j] = gdal.ApplyGeoTransform(transform_D, j, i)

                # EPSG:4326へ変換
                lons[i, j], lats[i, j] = transforme.transform(lons[i, j], lats[i, j])
                # 進捗バーを更新
                pbar_ac.update()
        pbar_ac.close()

        print("During adaptation to elevation data ...")

        # 緯度と経度をラジアンに変換
        L_rad = np.deg2rad(lats)
        B_rad = np.deg2rad(lons)

        # 楕円体の半径を計算
        R = R_equator * (1 - f * np.sin(L_rad)**2)**0.5

        # 楕円体の中心を原点としたときの各点の座標を計算
        X = R * np.cos(L_rad) * np.cos(B_rad)
        Y = R * np.cos(L_rad) * np.sin(B_rad)
        Z = R * np.sin(L_rad)

        # 楕円体を回転させるための回転行列を定義
        theta_L = np.pi / 2 - np.median(L_rad)  # 中心の緯度が最大になるようにθを設定
        theta_B = np.pi / 2 - np.median(B_rad)  # 中心の経度が最大になるようにθを設定
        rotation_matrix_L = np.array([
            [np.cos(theta_L), -np.sin(theta_L), 0],
            [np.sin(theta_L), np.cos(theta_L), 0],
            [0, 0, 1]
        ])
        rotation_matrix_B = np.array([
            [np.cos(theta_B), -np.sin(theta_B), 0],
            [0, 0, 1],
            [-np.sin(theta_B), np.cos(theta_B), 0]
        ])

        # 各点の座標を回転させる
        rotated_coords_L = np.dot(rotation_matrix_L, np.array([X.flatten(), Y.flatten(), Z.flatten()]))
        rotated_coords_B = np.dot(rotation_matrix_B, rotated_coords_L)
        rotated_X = rotated_coords_B[0, :].reshape(rows, cols)
        rotated_Y = rotated_coords_B[1, :].reshape(rows, cols)
        rotated_Z = rotated_coords_B[2, :].reshape(rows, cols)

        # 回転させた後の各点の高度を計算
        rotated_elevation = np.sqrt(rotated_X**2 + rotated_Y**2 + rotated_Z**2) - R

        #print(np.min((rotated_elevation - np.min(rotated_elevation)).reshape(rows, cols)))
        #print(np.max((rotated_elevation - np.min(rotated_elevation)).reshape(rows, cols)))

        # arrayに回転させた後の高度から最小値を引いた値を加算
        array += (rotated_elevation - np.min(rotated_elevation)).reshape(rows, cols)

    print("Done")
    print("Outputting png image...")
    # 正規化した標高データを基に色分けしたPNG画像を出力
    normalized_array = (((array - array.min()) / (array.max() - array.min())* 65535).astype(np.uint16))
    plt.imsave(os.path.join(output_dir, 'output.png'), (normalized_array), cmap='gray')

    print("Done")

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
        for res in landscape_res:
            # 分割後の画像を保存するディレクトリを作成
            res_dir = os.path.join(output_dir, f'resolution_{res}x{res}')
            os.makedirs(res_dir, exist_ok=True)

            # 画像を指定した解像度で分割し、余白部分を黒で塗りつぶす
            tiles, x_tiles, y_tiles = split_and_fill_image(array, normalized_array, res, small_units, transform_D, sigma)

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

    print("Complete")
