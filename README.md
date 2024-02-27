# MERGE GeoTIFF TOOLS

## EN

The software merges several GeoTIFF files present in a folder by sorting them into the correct arrangement.  
It can also export data for UE landscapes.


### Attention
**Do not install on paths where non-English characters are used.**  
May use huge amounts of memory.  
The programmer used up to 28 GB of memory when using the software with 9 (3 x 3) 5 km x 5 km , 1 M/px GeoTIFF tiles distributed by the '[National LIDAR Programme](https://www.data.gov.uk/dataset/f0db0249-f17b-4036-9e65-309148c97ce4/national-lidar-programme)'.  
Please use this as a guide.  

### Option Description.

#### --input-dir (required)
Directory containing the input GeoTIFF files.  
Example: --input-dir /path/to/tiff/folder  

#### --output-dir (required)  
Directory to save the output files.  
Example: --output-dir /path/to/output/folder  

#### --sigma  
Sigma value for the Gaussian blur. Default is 0 (no blur).  
Recommended to use when data is rough (large M/px values).  
Example: --sigma 3.5  

#### --output-graphs  
Output graphs for visual recognition of outliers. Can be very memory intensive.  
Example: --output-graphs  

#### --data-excluded  
Remove outliers from the data by removing the upper % and lower %. Default is 0 (calculated based on all data).  
Example: --data-excluded 0.05  

#### --data-excluded_u  
**NEVER USE WITH --data-excluded.**  
Remove outliers from the data by removing the upper %. Default is 0.  
Example: --data-excluded_u 0.05  

#### --data-excluded_l  
**NEVER USE WITH --data-excluded.**  
Remove outliers from the data by removing the lower %. Default is 0.  
Example: --data-excluded_l 0.05  

#### --not-flat-earth
Produce height map data along the spherical shape of the earth.  
Example: --not-flat-earth  

#### --ue-landscape  
Prepare data for Unreal Engine landscapes.  
Crop the image onto squares 127,253,505,1009,2017,4033,8129. This is the recommended resolution for height maps in landscape.  
Example: --ue-landscape  

#### --small-units  
**ALWAYS USE WITH --ue-landscape.**  
When binarising elevation data, binarise with elevation data for individual tiles.  
When importing very complex and very large maps, more detailed maps can be created as height map data is created for each small section of space.  
However, it is not necessary unless the map is very complex, as the height of each landscape break may vary slightly.  
Example: --ue-landscape --small-units  

## JA

このソフトウェアは、フォルダ内に存在する複数の GeoTIFF ファイルを正確な位置に配置し、結合します。  
また、Unreal Engine のランドスケープに利用するためのデータを書き出すことも可能です。


### 注意
**日本語など英語以外の文字が利用されている場所に設置しないでください。**  
このソフトウェアは大量のメモリを消費する可能性があります。  
製作者が実際に '[National LIDAR Programme](https://www.data.gov.uk/dataset/f0db0249-f17b-4036-9e65-309148c97ce4/national-lidar-programme)' からダウンロードした 5km x 5km で 1M/px の GeoTIFF ファイル 9(3x3)枚を結合しようとした際、最大28GBのメモリを消費しました。  
利用時の目安として参考にしてください。  

### オプションの詳細


#### --input-dir (必須)
GeoTIFF ファイルが入っているフォルダを指定してください。  
例: --input-dir /path/to/tiff/folder  

#### --output-dir (必須)
出力先のフォルダを指定してください。  
例: --output-dir /path/to/output/folder  

#### --sigma
画像ファイルにかけるブラーの値を指定してください。デフォルトは 0 (ブラー無し)です。  
特に、M/px の値が大きい(データが荒い)際に利用することを推奨します。  
例: --sigma 3.5  

#### --output-graphs
外れ値を可視化しやすくするためにグラフを出力します。大量のメモリを消費する可能性があります。  
例: --output-graphs  

#### --data-excluded
外れ値を取り除くため、上位下位それぞれ何%のデータを取り除くか指定します。デフォルトは 0 です(すべてのデータを利用します)。  
例: --data-excluded 0.05  

#### --data-excluded_u
**絶対に --data-excluded と併用しないでください。**  
外れ値を取り除くため、上位何%のデータを取り除くか指定します。デフォルトは 0 です。  
例: --data-excluded_u 0.05  

#### --data-excluded_l
**絶対に --data-excluded と併用しないでください。**  
外れ値を取り除くため、下位何%のデータを取り除くか指定します。デフォルトは 0 です。  
例: --data-excluded_l 0.05  

#### --not-flat-earth
地球の球形に沿った高さマップデータを作成します。  
例: --not-flat-earth  

#### --ue-landscape
Unreal Engine でのランドスケープ向けにデータを書き出します。  
その際に、高さマップ画像を 127,253,505,1009,2017,4033,8129 四方にそれぞれ分割します。これはUEのランドスケープで読み込む高さマップの推奨値です。  
例: --ue-landscape  

#### --small-units
**必ず --ue-landscape と併用してください。**  
タイルごとの高度データをもとにして二値化したデータを出力します。  
非常に複雑で大きな地図をインポートする場合、空間の小さなセクションごとに高さマップデータが作成されるため、より詳細なデータを作成することができます。  
正し、ランドスケープごとの高さデータが若干ずれる可能性があるので、非常に複雑で大きな地図をインポートする場合以外は推奨しません。  
例: --ue-landscape --small-units  

## This software uses the following licence packages
GDAL:  MIT License  
Nuitka:  Apache Software License  
argparse:  Python Software Foundation License  
certifi:  Mozilla Public License 2.0 (MPL 2.0)  
contourpy:  BSD License  
cycler:  BSD License  
delphifmx:  Other/Proprietary License (BSD)  
fonttools:  MIT License  
kiwisolver:  BSD License  
matplotlib:  Python Software Foundation License  
numpy:  BSD License  
ordered-set:  MIT License  
packaging:  Apache Software License; BSD License  
pandas:  BSD License  
pillow:  Historical Permission Notice and Disclaimer (HPND)  
pyparsing:  MIT License  
python-dateutil:  Apache Software License; BSD License  
pytz:  MIT License  
scipy:  BSD License
seaborn:  BSD License
six:  MIT License
tzdata:  Apache Software License
zstandard: BSD License
