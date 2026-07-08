import os
import shutil
import subprocess
import sys
from pathlib import Path

from delphifmx import *
from . import icon

#ファイルダイアログ等専用
from tkinter import Tk
from tkinter import filedialog as fd
from tkinter import messagebox as msgb
#Tkinterのメインウィンドウ隠すための処理
fdw = Tk()
fdw.withdraw()
fdw.iconphoto(True, icon.icon())

#import Merge_GeoTIFF as mtg

PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_LANDSCAPE_RESOLUTIONS_TEXT = '127 253 505 1009 2017 4033 8129'


def resolve_cli_command():
    for base_dir in (Path.cwd(), SOURCE_ROOT, PACKAGE_DIR):
        exe_path = base_dir / 'Merge_GeoTIFF.exe'
        if exe_path.is_file():
            return [str(exe_path)]

    installed_command = shutil.which('merge-geotiff') or shutil.which('Merge_GeoTIFF.exe')
    if installed_command:
        return [installed_command]

    script_path = SOURCE_ROOT / 'Merge_GeoTIFF.py'
    if script_path.is_file():
        return [sys.executable, str(script_path)]

    return None


def parse_landscape_resolution_args(value):
    resolutions = str(value).split()
    if not resolutions:
        raise ValueError('Landscape resolution is empty.')
    for resolution in resolutions:
        if int(resolution) <= 0:
            raise ValueError('Landscape resolution must be positive.')
    return resolutions

class main(Form):

    #実行
    def execution(self):
        #フォルダのチェック
        if os.path.isdir(str(self.Input_TextBox.Text)) == False:
            msgb.showerror('The input directory does not exist.', 'Please specify the input directory to "Set Input dir here."')
            return
        if os.path.isdir(str(self.Output_TextBox.Text)) == False:
            msgb.showerror('The output directory does not exist.', 'Please specify the output directory to "Set Output dir here."')
            return
        input_dir = str(self.Input_TextBox.Text)
        output_dir = str(self.Output_TextBox.Text)
        try:
            sigma = float(self.Sigma_TextBox.Text)
            data_excluded = float(self.Data_excluded_TextBox.Text)
            data_excluded_u = float(self.Data_excluded_u_TextBox.Text)
            data_excluded_l = float(self.Data_excluded_l_TextBox.Text)
        except ValueError:
            msgb.showerror('Invalid number.', 'Sigma and data exclusion values must be numbers.')
            return
        output_graphs = self.Output_graphs.IsChecked
        not_flat_earth = self.Not_Flat_Earth.IsChecked
        ue_landscape = self.UE_Landscape.IsChecked
        small_units = self.Small_Units.IsChecked

        cli_command = resolve_cli_command()
        if cli_command is None:
            msgb.showerror('"Merge_GeoTIFF" is missing.', 'Please run the GUI in the directory where "Merge_GeoTIFF.exe" or "Merge_GeoTIFF.py" exists.')
            return

        command = [
            *cli_command,
            '--input-dir', input_dir,
            '--output-dir', output_dir,
            '--sigma', str(sigma),
            '--data-excluded', str(data_excluded),
            '--data-excluded_u', str(data_excluded_u),
            '--data-excluded_l', str(data_excluded_l),
        ]
        if output_graphs:
            command.append('--output-graphs')
        if not_flat_earth:
            command.append('--not-flat-earth')
        if ue_landscape:
            try:
                landscape_resolutions = parse_landscape_resolution_args(self.Landscape_resolution_Textbox.Text)
            except ValueError:
                msgb.showerror('Invalid landscape resolution.', 'Landscape resolutions must be positive integers separated by spaces.')
                return
            command.append('--ue-landscape')
            command.append('--landscape-res')
            command.extend(landscape_resolutions)
        if small_units:
            command.append('--small-units')

        creationflags = getattr(subprocess, 'CREATE_NEW_CONSOLE', 0) if os.name == 'nt' else 0
        subprocess.Popen(command, cwd=str(Path.cwd()), creationflags=creationflags)
        #subprocess.run(['Merge_GeoTIFF.exe', f'--input-dir "{input_dir}"', f'--output-dir "{output_dir}"', f'--sigma {sigma}', f'{output_graphs}', f'--data-excluded {data_excluded}', f'--data-excluded_u {data_excluded_u}', f'--data-excluded_l {data_excluded_l}', f'{ue_landscape}', f'{small_units}'])
        #mtg.main(f"{input_dir}",f"{output_dir}",sigma,output_graphs,data_excluded,data_excluded_u,data_excluded_l,ue_landscape,small_units)

    #初期化処理
    def __init__(self, owner):
        self.Input = None
        self.Input_TextBox = None
        self.Input_Button = None
        self.Output = None
        self.Output_TextBox = None
        self.Output_Button = None
        self.Option = None
        self.Output_graphs = None
        self.Sigma = None
        self.Sigma_TextBox = None
        self.Data_excluded = None
        self.Data_excluded_TextBox = None
        self.Data_excluded_l = None
        self.Data_excluded_l_TextBox = None
        self.Data_excluded_u = None
        self.Data_excluded_u_TextBox = None
        self.UE_Landscape = None
        self.Small_Units = None
        self.Select_Data_excluded = None
        self.Normal = None
        self.U_L = None
        self.Not_Flat_Earth = None
        self.Run = None
        self.Reset = None
        self.Landscape_resolution = None
        self.Landscape_resolution_Textbox = None
        self.LoadProps(str(PACKAGE_DIR / "Merge_GeoTIFF_GUI.pyfmx"))

    def Input_ButtonClick(self, Sender):
        self.Input_TextBox.text = str(fd.askdirectory(parent = fdw))

    def Output_ButtonClick(self, Sender):
        self.Output_TextBox.text = str(fd.askdirectory(parent = fdw))

    def UE_LandscapeChange(self, Sender):
        if self.Small_Units.Enabled == False:
            self.Small_Units.Enabled = True
            self.Landscape_resolution_Textbox.Enabled = True
        else:
            self.Small_Units.IsChecked = False
            self.Small_Units.Enabled = False
            self.Landscape_resolution_Textbox.Enabled = False

    def Select_Data_excludedChange(self, Sender):
        if self.Select_Data_excluded.ItemIndex == 0:
            self.Data_excluded_TextBox.Enabled = True
            self.Data_excluded_u_TextBox.Enabled = False
            self.Data_excluded_u_TextBox.Text = 0.0
            self.Data_excluded_l_TextBox.Enabled = False
            self.Data_excluded_l_TextBox.Text = 0.0
        else:
            self.Data_excluded_TextBox.Enabled = False
            self.Data_excluded_TextBox.Text = 0.0
            self.Data_excluded_u_TextBox.Enabled = True
            self.Data_excluded_l_TextBox.Enabled = True

    def RunClick(self, Sender):
        self.execution()

    def ResetClick(self, Sender):
        #入出力フォルダ
        self.Input_TextBox.Text = ''
        self.Output_TextBox.Text = ''
        #ガウスぼかし
        self.Sigma_TextBox.Text = 0.0
        #外れ値の除外
        self.Select_Data_excluded.ItemIndex = 0
        self.Data_excluded_TextBox.Enabled = True
        self.Data_excluded_TextBox.Text = 0.0
        self.Data_excluded_l_TextBox.Enabled = False
        self.Data_excluded_l_TextBox.Text = 0.0
        self.Data_excluded_u_TextBox.Enabled = False
        self.Data_excluded_u_TextBox.Text = 0.0
        #グラフ
        self.Output_graphs.IsChecked = False
        #ランドスケープ
        self.UE_Landscape.IsChecked = False
        self.Small_Units.IsChecked = False
        self.Small_Units.Enabled = False
        self.Landscape_resolution_Textbox.Enabled = False
        self.Landscape_resolution_Textbox.Text = DEFAULT_LANDSCAPE_RESOLUTIONS_TEXT
        #フラットアーサー
        self.Not_Flat_Earth.IsChecked = False

        msgb.showinfo('Reset','All settings have been returned to their initial state.')

def main_func():

    if resolve_cli_command() is None:
        msgb.showerror('"Merge_GeoTIFF" is missing.','\nPlease run the GUI in the directory where "Merge_GeoTIFF.exe" or "Merge_GeoTIFF.py" exists.')
        return

    Application.Initialize()
    Application.Title = 'Merge GeoTIFF GUI'
    Application.MainForm = main(Application)
    Application.MainForm.Show()
    Application.Run()
    Application.MainForm.Destroy()

if __name__ == '__main__':
    main_func()
