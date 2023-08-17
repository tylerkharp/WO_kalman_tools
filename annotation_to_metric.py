import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import annotation_manipulation as am
import matplotlib.colors as mcolors



def get_error_metrics(name, annotations_list,expected_elevation,encoder):
    """illustrate how closely the annotations match the expected elevation"""
    
    fig = plt.figure(figsize=(20,8))
    gs = fig.add_gridspec(1, 2, width_ratios=(3, 1))
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    padding = 0.15
    if len(annotations_list) == 1:
        annotations = annotations_list[0]    
        if expected_elevation == 0:
            encoder_error_ratio = 0.5/40 #ft/ft
            encoder_max_elevation = annotations.Y.max()
            expected_elevation = encoder_max_elevation / (1-encoder_error_ratio)
        average_error = (annotations.Y - expected_elevation).mean()
        average_std = (annotations.Y - expected_elevation).std()
        y_lim_max = expected_elevation + padding
        y_lim_min = annotations.Y.min() - padding

        ax1.stem(
            annotations.X.astype(float),
            annotations.Y,
            mcolors.CSS4_COLORS['cornflowerblue'],
            bottom=expected_elevation,
            label=f"{encoder} Encoder Weld Line Mapping",
        )
        ax1.set_title(name)
        ax1.set_xlabel("Asset horizontal position (in.)")
        ax1.set_ylabel("Apparent elevation (ft.)")
        ax1.legend()
        ax1.set_ylim((y_lim_min, y_lim_max))

        corr = annotations.Y - expected_elevation
        ax2.hist(corr, bins=150, color = mcolors.CSS4_COLORS['cornflowerblue'], range=(y_lim_min-expected_elevation,y_lim_max-expected_elevation))
        ax2.set_title(f"Distribution of {encoder} Encoder to expected elevation")
        ax2.set_ylabel("Count")
        ax2.set_xlabel("Difference (ft.)")
    else:
        left_annotations = annotations_list[0]
        external_annotations = annotations_list[1]    
        average_error = (external_annotations.Y - expected_elevation).mean()
        average_std = (external_annotations.Y - expected_elevation).std()
        y_lim_max = max(external_annotations.Y.max(),expected_elevation) + padding
        y_lim_min = min(external_annotations.Y.min(),left_annotations.Y.min()) - padding
        ax1.stem(
            left_annotations.X.astype(float),
            left_annotations.Y,
            mcolors.CSS4_COLORS['lightsteelblue'],
            bottom=expected_elevation,
            label=f"{encoder} Encoder Weld Line Mapping",
        )
        ax1.stem(
            external_annotations.X.astype(float),
            external_annotations.Y,
            mcolors.CSS4_COLORS['red'],
            bottom=expected_elevation,
            label="Kalman Encoder Weld Line Mapping",
        )
        ax1.set_title(name + f" of Kalman Filter versus {encoder} encoder")
        ax1.set_xlabel("Asset horizontal position (in.)")
        ax1.set_ylabel("Apparent elevation (ft.)")
        ax1.legend()
        ax1.set_ylim((y_lim_min, y_lim_max))

        external_corr = external_annotations.Y - expected_elevation
        left_corr = left_annotations.Y - expected_elevation
        ax2.hist(left_corr, bins=150, color = mcolors.CSS4_COLORS['lightsteelblue'] , lw=0, range=(y_lim_min-expected_elevation,y_lim_max-expected_elevation))
        ax2.hist(external_corr, bins=150, color = mcolors.CSS4_COLORS['red'] , lw=0, range=(y_lim_min-expected_elevation,y_lim_max-expected_elevation))

        ax2.set_title(f"Distribution of Kalman Encoder versus {encoder} encoder to expected elevation")
        ax2.set_ylabel("Count")
        ax2.set_xlabel("Difference (ft.)")
    
    return average_error, average_std, expected_elevation


def split_annotations_into_weld_lines(
    df
):
    number_of_weld_lines = 2
    weld_line_df_list = {}
    if "orientation" in df.columns:
        df = df[df["orientation"] == 1]
    if len(df.X.unique()) == 1:
        df.X = np.linspace(0, 1, len(df.X))    
    if 'X' and 'elevation' in df.columns:
        plt.scatter(df.X, df.elevation)
    else:
        plt.scatter(df.iloc[:,6], df.iloc[:,2])
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    x = plt.ginput(number_of_weld_lines * 2, timeout=60)
    plt.close()
    x = np.array(x)
    bins = []
    for coords in x:
        bins.append(coords[1])
    bins = sorted(bins)
    df["weld_line"] = pd.cut(df["elevation"], bins=bins, labels=False)
    for i in range(number_of_weld_lines):
        weld_line_df = df[df["weld_line"] == i * 2]
        duplicated_or_not_df = weld_line_df.duplicated(subset=["run_name"])
        weld_line_df = weld_line_df[~duplicated_or_not_df]
        weld_line_df = weld_line_df.sort_values(by=["run_name"])
        weld_line_df = weld_line_df.reset_index(drop=True)
        weld_line_df_list["weld line #" + str(i + 1)] = weld_line_df
    return weld_line_df_list


def get_run_offsets(first_weld_line_df_list):
    offset_list = {}
    average_first_weld_line_height = first_weld_line_df_list["elevation"].mean()
    for run_name in first_weld_line_df_list["run_name"]:
        weld_line_start = first_weld_line_df_list[
            first_weld_line_df_list["run_name"] == run_name
        ]["elevation"]
        assert len(weld_line_start) == 1
        offset = average_first_weld_line_height - float(weld_line_start)
        offset_list[run_name] = offset
    return offset_list


def apply_offsets_to_weld_lines(weld_line_df_list, offset_list):
    weld_line_offset_list = {}
    for weld_line in weld_line_df_list:
        weld_line_df = weld_line_df_list[weld_line]
        offset_df = pd.DataFrame(columns=["run_name", "X", "Y"])
        for run_name in weld_line_df_list[weld_line]["run_name"]:
            if run_name in offset_list:
                old_row = weld_line_df[weld_line_df["run_name"] == run_name]
                old_value = float(old_row["elevation"])
                offset = float(offset_list[run_name])
                new_value = old_value + offset
                offset_df.loc[len(offset_df.index)] = [run_name,old_row['X'],new_value]
        weld_line_offset_list[weld_line] = offset_df
    return weld_line_offset_list

def get_and_offset_weld_lines(slug_name, encoder):
    df = am.load_annotations_as_df(f"{slug_name}/{slug_name}_{encoder}_annotations.pkl")
    a = pd.concat([df.run_name, df.sensor_index,df.ascan_index], axis=1)
    weld_line_df_list = split_annotations_into_weld_lines(df)
    first_weld_line_df_list = weld_line_df_list["weld line #1"]
    am.save_var_as_pickle(first_weld_line_df_list, f"{slug_name}/{encoder}_first_weld_line")
    offset_list = get_run_offsets(first_weld_line_df_list)
    print(offset_list)
    weld_line_offset_list = apply_offsets_to_weld_lines(weld_line_df_list, offset_list)
    plt.scatter(weld_line_offset_list["weld line #1"].X, weld_line_offset_list["weld line #1"].Y)
    plt.scatter(weld_line_offset_list["weld line #2"].X, weld_line_offset_list["weld line #2"].Y)
    plt.show()
    return weld_line_offset_list

def get_final_metrics(slug_name, encoder,compare_right_encoder):
    slug_folder = f"{slug_name}"
    if not os.path.exists(f"{slug_folder}"):
        os.mkdir(slug_folder)
    elevation_file_list = os.listdir("elevations")
    for file in elevation_file_list:
        if slug_name in file:
            if not os.path.exists(f"{slug_folder}/{file}"):
                os.rename(f"elevations/{file}", f"{slug_folder}/{file}")
            else:
                os.remove(f"{slug_folder}/{file}")
                os.rename(f"elevations/{file}", f"{slug_folder}/{file}")      
    if encoder == "left":
        weld_line_offsetted_list = get_and_offset_weld_lines(slug_name,encoder="left")
        am.save_var_as_pickle(weld_line_offsetted_list, f"{slug_folder}/{slug_name}_left_offsetted_annotations")
        average_error, average_std, estimated_height = get_error_metrics("Weld Line #2", [weld_line_offsetted_list["weld line #2"]],0,encoder)
        plt.savefig(slug_folder + "/" + slug_name + "_left_error_metrics.png")
        np.save(slug_folder + "/" + slug_name + "_left_error_metrics.npy", [average_error, average_std, estimated_height])
    if encoder == "right":
        right_weld_line_offsetted_list = get_and_offset_weld_lines(slug_name,encoder="right")
        left_error_metrics = np.load(slug_folder + "/" + slug_name + "_left_error_metrics.npy")
        am.save_var_as_pickle(right_weld_line_offsetted_list, f"{slug_folder}/{slug_name}_right_offsetted_annotations")
        average_error, average_std, estimated_height = get_error_metrics("Weld Line #2", [right_weld_line_offsetted_list["weld line #2"]],left_error_metrics[2],encoder)
        plt.savefig(slug_folder + "/" + slug_name + "_right_error_metrics.png")
        np.save(slug_folder + "/" + slug_name + "_right_error_metrics.npy", [average_error, average_std, estimated_height])
    if encoder == "external":

        external_weld_line_offsetted_list = get_and_offset_weld_lines(slug_name, encoder="external")
        print(external_weld_line_offsetted_list['weld line #1']['Y'])
        left_error_metrics = np.load(slug_folder + "/" + slug_name + "_left_error_metrics.npy")
        if not compare_right_encoder:
            left_weld_line_offsetted_list = get_and_offset_weld_lines(slug_name,encoder="left")
            average_error, average_std, estimated_height = get_error_metrics("Weld Line #2", [left_weld_line_offsetted_list["weld line #2"],external_weld_line_offsetted_list["weld line #2"]], left_error_metrics[2],"left")
            plt.savefig(slug_folder + "/" + slug_name + "_external_error_metrics.png",dpi=300)
            np.save(slug_folder + "/" + slug_name + "_external_error_metrics.npy", [average_error, average_std, estimated_height])
        if compare_right_encoder:
            right_weld_line_offsetted_list = get_and_offset_weld_lines(slug_name,encoder="right")
            average_error, average_std, estimated_height = get_error_metrics("Weld Line #2", [right_weld_line_offsetted_list["weld line #2"],external_weld_line_offsetted_list["weld line #2"]], left_error_metrics[2],"right")
            plt.savefig(slug_folder + "/" + slug_name + "_external_versus_right.png",dpi=300)
            np.save(slug_folder + "/" + slug_name + "_external_error_metrics.npy", [average_error, average_std, estimated_height])

def load_and_print_error_metrics(slug_name, encoder):
    error_metrics = np.load(slug_name + "/" + slug_name + "_" + encoder + "_error_metrics.npy")
    print("Encoder: " + encoder)
    print("Average Error: " + str(error_metrics[0] * 12) + "in")
    print("Average Std: " + str(error_metrics[1]* 12) + "in")
    print("Estimated Height: " + str(error_metrics[2]) + "ft")

def assemble_all_error_metrics():
    all_error_metrics = np.zeros((18,5))
    all_error_metrics = pd.DataFrame(all_error_metrics,columns=["Slug Name","Encoder","Average Error","Average Std","Expected Height"])
    count = 0
    for folder in os.listdir():
        if os.path.isdir(folder) and "2023" in folder:
            error_metrics = np.zeros((3,5))
            error_metrics = pd.DataFrame(error_metrics,columns=["Slug Name","Encoder","Average Error","Average Std","Expected Height"])
            error_metrics.iloc[0,1] = "Left"
            error_metrics.iloc[1,1] = "Right"
            error_metrics.iloc[2,1] = "External"
            error_metrics.iloc[:,0] = folder
            if os.path.exists(folder + "/" + folder + "_left_error_metrics.npy"):
                left_metrics = np.load(folder + "/" + folder + "_left_error_metrics.npy")
            else:
                left_metrics = np.zeros(3)
            if os.path.exists(folder + "/" + folder + "_external_error_metrics.npy"):
                external_metrics = np.load(folder + "/" + folder + "_external_error_metrics.npy")
            else:
                external_metrics = np.zeros(3)
            if os.path.exists(folder + "/" + folder + "_right_error_metrics.npy"):
                right_metrics = np.load(folder + "/" + folder + "_right_error_metrics.npy")
            else:
                right_metrics = np.zeros(3)
            error_metrics.iloc[0,2:5] = left_metrics
            error_metrics.iloc[1,2:5] = right_metrics
            error_metrics.iloc[2,2:5] = external_metrics
            all_error_metrics.iloc[count*3:count*3+3,:] = error_metrics
            count += 1
    all_error_metrics.to_csv("all_error_metrics.csv")

if __name__ == "__main__":
    slug_list = {
            # "20230131-7e2700":False,
            # "20230615-086467":True,
            # "20230508-04487e":True,
            # "20230516-3c4686":True,
            # "20230427-e92647":False,
            # "20230522-1ac0e6":False,
            }
    slug_name = "20230522-1ac0e6"
    get_final_metrics(slug_name, "external",compare_right_encoder=False)
    plt.show()
    # load_and_print_error_metrics(slug_name, "external")
    # assemble_all_error_metrics()


