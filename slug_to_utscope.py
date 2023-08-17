import google.cloud.storage as gcs
import os
import numpy as np
import pandas as pd
import json
import terrarium_utils as tu
import ObjDataManipulation as odm
import shutil

class SlugData:
    terrarium_event_file_list = []
    terrarium_log_file_list = []
    terrarium_file_list = []
    Q_set = 0.8488891111
    Rm_set = 0.822222
    Rwb_set = 0.4405555556
    gscale_set = 1.031066
    vel_offset = 0.20
    GT_offset = 0
    height_offset = 0
    used_column_list = ['/timestamp',
                        '/left_drive_outputs/encoder_pos_count',
                        '/right_drive_outputs/encoder_pos_count',
                        '/left_drive_outputs/axis0/pos_estimate_count',
                        '/left_drive_outputs/axis1/pos_estimate_count',
                        '/right_drive_outputs/axis0/pos_estimate_count',
                        '/right_drive_outputs/axis1/pos_estimate_count',
                        '/left_drive_inputs/velocity_cmd',
                        '/right_drive_inputs/velocity_cmd',
                        '/brain_outputs/robot_position_ft',]
    used_column_list_with_inputs = ['/timestamp',
                        '/inputs/left_drive_outputs/encoder_pos_count',
                        '/inputs/right_drive_outputs/encoder_pos_count',
                        '/inputs/left_drive_outputs/axis0/pos_estimate_count',
                        '/inputs/left_drive_outputs/axis1/pos_estimate_count',
                        '/inputs/right_drive_outputs/axis0/pos_estimate_count',
                        '/inputs/right_drive_outputs/axis1/pos_estimate_count',
                        '/inputs/left_drive_inputs/velocity_cmd',
                        '/inputs/right_drive_inputs/velocity_cmd',
                        '/inputs/brain_outputs/robot_position_ft',]

    def __init__(self,slug_name: str, directory: str, is_data_already_there: bool,apply_operator_offset: bool):
        self.apply_operator_offset = apply_operator_offset
        self.slug_name = slug_name
        self.directory = directory
        self.create_slug_folders()
        self.get_history_files(is_data_already_there)
        self.get_history_events()
        self.get_and_save_terrarium_files(is_data_already_there)
        self.check_slug_validity()
        self.merge_all_log_files()


    def break_up_into_runs(self):
        self.get_terrarium_events()
        for computer_name in self.history_computer_names:
            self.synchronize_log_and_history_clocks(computer_name)
            self.split_data_into_runs(computer_name)

    def create_slug_folders(self):
        folder_path = self.directory + '/' + self.slug_name
        self.slug_folder_path = folder_path
        self.history_folder_path = self.slug_folder_path + '/history'
        self.terrarium_folder_path = self.slug_folder_path + '/terrarium_files'
        self.split_terrarium_data_folder_path = self.slug_folder_path + '/raw_run_data'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(self.history_folder_path):
            os.makedirs(self.history_folder_path)
        if not os.path.exists(self.terrarium_folder_path):
            os.makedirs(self.terrarium_folder_path)
        if not os.path.exists(self.split_terrarium_data_folder_path):
            os.mkdir(self.split_terrarium_data_folder_path)
        self.create_ut_scope_folder()
        
    def create_ut_scope_folder(self):
        path = self.slug_folder_path + '/' + 'UTScope_final_data'
        if not os.path.exists(path):
            os.makedirs(path)
        utscope_data_path = path + '/data'
        utscope_plots_path = path + '/plots'
        if not os.path.exists(utscope_data_path):
            os.makedirs(utscope_data_path)
        if not os.path.exists(utscope_plots_path):
            os.makedirs(utscope_plots_path)
        self.utscope_data_path = utscope_data_path
        self.utscope_plots_path = utscope_plots_path
        

    def get_history_files(self,is_data_already_there: bool):
        if not is_data_already_there:
            storage_client = gcs.Client()
            self.bucket = storage_client.get_bucket('gecko-archive')
            slug_blob_list = list(self.bucket.list_blobs(prefix=f"{self.slug_name}/"))
            self.history_file_list = []
            for blob in slug_blob_list:
                if "history.json" in blob.name and not ("upload_history.json" in blob.name):
                    file_path = self.history_folder_path + '/' + blob.name.replace('/', '_') 
                    self.history_file_list.append(file_path)
                    blob.download_to_filename(file_path)
            if self.history_file_list == []:
                print("No History files found for this slug")
            print("GC History files saved")
        else:
            file_list = os.listdir(self.history_folder_path) 
            self.history_file_list = [self.history_folder_path +'/'+ file_name for file_name in file_list]
            print("GC History files already saved")

    def get_and_save_terrarium_files(self,is_data_already_there: bool):
        if not is_data_already_there:
            self.get_computer_names()
            self.slug_info = {'computer_list':self.history_computer_names,'start_date':self.inspection_start,'end_date':self.inspection_end}
            log_content_list,event_content_list = tu.get_slug_files_from_terrarium(self.slug_name,self.slug_info,authentication_directory='/home/tyler.harp/UTScopeData/authentication')
            self.terrarium_event_file_list = []
            self.terrarium_log_file_list = []
            if log_content_list == {} and event_content_list == {}:
                print("No Terrarium files found for this slug")
            else:
                for key in log_content_list:
                    file_name = key.replace('/','_')
                    self.terrarium_log_file_list.append(file_name+'.csv')
                    with open (self.terrarium_folder_path + '/' + f"{file_name}.csv", "w") as f:
                        f.write(log_content_list[key])
                for key in event_content_list:
                    file_name = key.replace('/','_')
                    self.terrarium_event_file_list.append(file_name)
                    with open (self.terrarium_folder_path + '/' + f"{file_name}", "w") as f:
                        f.write(event_content_list[key])
                print("All Terrarium files saved")
        else:
            self.terrarium_file_list = os.listdir(self.terrarium_folder_path)
            for file_name in self.terrarium_file_list:
                if 'csv' in file_name:
                    self.terrarium_log_file_list.append(file_name)
                else:
                    self.terrarium_event_file_list.append(file_name)
            print("Terrarium files already saved")

    def get_computer_names(self):
        if not self.history_file_list == []:        
            number_of_files = len(self.history_file_list)
            self.history_computer_names = []
            for i in range(number_of_files):
                second_computer_name_index = self.history_file_list[i].rfind('_')
                first_computer_name_index = self.history_file_list[i].rfind('_',0,second_computer_name_index)
                self.history_computer_names.append(self.history_file_list[i][first_computer_name_index+1:second_computer_name_index])
            self.history_computer_names = list(set(self.history_computer_names))
        else:
            print('No History files to get computer names from')
        if not self.terrarium_event_file_list == []:
            self.terrarium_computer_names = []
            for i in range(len(self.terrarium_event_file_list)):
                computer_name_start_index = self.terrarium_event_file_list[i].find('-')
                computer_name_end_index = self.terrarium_event_file_list[i].find('_')
                self.terrarium_computer_names.append(self.terrarium_event_file_list[i][computer_name_start_index+1:computer_name_end_index])
            self.terrarium_computer_names = list(set(self.terrarium_computer_names))

    def check_slug_validity(self):
        self.get_computer_names()
        self.get_asset_serials()
        for computer_name in self.terrarium_asset_serials:
            if not len(self.terrarium_asset_serials[computer_name]) == 1:
                exit("More than one asset serial number found for " + computer_name)

        if not len(self.terrarium_computer_names) == len(self.history_computer_names):
            exit("Number of computer names in history and terrarium do not match")

    def get_asset_serials(self):
        if not self.terrarium_event_file_list == []:
            self.terrarium_asset_serials = {}
            for file_name in self.terrarium_event_file_list:
                #get the 'CB400000031' out of 'gecko-g4-0088_ekko_logs_CB400000031__2023-02-08_04-11-46.events.log'
                asset_serial_index = file_name.rfind('logs_')
                asset_serial_second_index = file_name.rfind('__')
                asset_serial = file_name[asset_serial_index+5:asset_serial_second_index]
                computer_name_end_index = file_name.find('_')
                computer_name = file_name[6:computer_name_end_index]

                if computer_name in self.terrarium_asset_serials:
                    if asset_serial not in self.terrarium_asset_serials[computer_name]:
                        self.terrarium_asset_serials[computer_name].append(file_name[asset_serial_index+5:asset_serial_second_index])
                else:
                    self.terrarium_asset_serials.update({computer_name:[file_name[asset_serial_index+5:asset_serial_second_index]]})

    def get_history_events(self):
        self.start_history_events_list = {}
        self.history_events_list = {}
        for file_path in self.history_file_list:
            computer_name_second_index = file_path.rfind('_')
            computer_name_index = file_path.rfind('_',0,computer_name_second_index)
            computer_name = file_path[computer_name_index+1:computer_name_second_index]
            history = pd.DataFrame(json.load(open(file_path, "r"))['history'])
            self.inspection_start = convert_timestamp_to_date(history['timestamp'][0])
            self.inspection_end = convert_timestamp_to_date(history['timestamp'][len(history)-1])
            column_list = history.columns
            for column in column_list:
                if 'run_id' in column or 'run_number' in column:
                    self.run_column_name = column
                    run_column_name = column
            history = history.dropna(subset=[run_column_name]).reset_index(drop=True)
            history = history[['type','timestamp',run_column_name]]
            # history = history.drop(['verification_skipped','sensor_id','channel'], axis=1)
            history['timestamp'] = pd.to_datetime(history['timestamp']).astype(int) / 10**9
            history[run_column_name] = history[run_column_name].astype(int)
            
            start_history = history[(history['type']=='RUN_START') | (history['type']=='run')]
            run_history = history[(history['type']=='RUN_START') | (history['type']=='run') | (history['type']=='RUN_COMPLETE')]
            start_history = start_history.rename(columns={run_column_name:'run_number'})
            run_history = run_history.rename(columns={run_column_name:'run_number'})

            
            self.history_events_list.update({computer_name:run_history})
            self.start_history_events_list.update({computer_name:start_history})

    def get_terrarium_events(self):
        self.start_terrarium_events_list = {}
        self.terrarium_events_list = {}
        for file_name in self.terrarium_event_file_list:
            computer_start_index = file_name.find('-')
            computer_end_index = file_name.find('_')
            computer_name = file_name[computer_start_index+1:computer_end_index]
            
            computer_name = computer_name.upper()
            with open(self.terrarium_folder_path + '/' + file_name, "r") as file:
                try:
                    events = [json.loads(line) for line in file.readlines()][1:]
                except:
                    exit(f"File ({file_name}) is corrupted")
            # Convert the list of events into a DataFrame
            data = {
                "/timestamp": [event["timestamp"] for event in events],
                "event_type": [event["event_type"] for event in events],
                "run_number": [event["event_data"] for event in events] 
            }
            events = pd.DataFrame.from_records(data=data)
            run_events = events[(events['event_type']=='inspection_start') | (events['event_type']=='inspection_stop')].reset_index(drop=True)
            start_events = events[events['event_type']=='inspection_start'].reset_index(drop=True)
            start_events['run_number'] = start_events['run_number'].apply(self.get_run_id_from_terrarium_event)
            run_events['run_number'] = run_events['run_number'].apply(self.get_run_id_from_terrarium_event)
            if computer_name in self.terrarium_events_list:
                self.terrarium_events_list[computer_name] = pd.concat([self.terrarium_events_list[computer_name],run_events]).reset_index(drop=True)
                self.start_terrarium_events_list[computer_name] = pd.concat([self.start_terrarium_events_list[computer_name],start_events]).reset_index(drop=True)
            else:
                self.start_terrarium_events_list.update({computer_name:start_events})
                self.terrarium_events_list.update({computer_name:run_events})


    def get_run_id_from_terrarium_event(self,event_data):
        run_id = int(event_data['run_data']['folder'][-4:])
        return run_id
    
    def synchronize_log_and_history_clocks(self,computer_name):
        start_history_events = self.start_history_events_list[computer_name]
        history_events = self.history_events_list[computer_name]
        terrarium_events = self.start_terrarium_events_list[computer_name]
        time_diffs = self.get_all_time_differences(terrarium_events,start_history_events) 
        avg_time_difference = float(sum(time_diffs) / len(time_diffs)) #if negative history is behind, if positive history is ahead
        if (time_diffs - avg_time_difference > 50).any():
            exit("Time difference between terrarium and history is too large")
        self.history_events_list[computer_name]['timestamp'] = history_events['timestamp'] - avg_time_difference
        self.start_history_events_list[computer_name]['timestamp'] = start_history_events['timestamp'] - avg_time_difference

    
    def get_all_time_differences(self,terrarium_events,history_events):
        time_diffs = np.zeros((len(terrarium_events),1))
        for i in range(len(terrarium_events)):
            history_sync_event = history_events[(history_events['run_number']==terrarium_events['run_number'][i]) & ((history_events['type']=='RUN_START') | (history_events['type']=='run'))]
            if history_sync_event.empty:
                exit("History file is incomplete")
            time_diff = history_sync_event['timestamp'].iloc[0] - terrarium_events['/timestamp'].iloc[i] #if negative history is behind, if positive history is ahead
            time_diffs[i] = time_diff
        return time_diffs
    
    def merge_all_log_files(self):
        self.all_log_data = {}
        for computer_name in self.terrarium_computer_names:
            data_list = []
            for i in range(len(self.terrarium_log_file_list)):
                computer_name_for_file_start_index = self.terrarium_log_file_list[i].find('-')
                computer_name_for_file_end_index = self.terrarium_log_file_list[i].find('_')
                computer_name_for_file = self.terrarium_log_file_list[i][computer_name_for_file_start_index+1:computer_name_for_file_end_index]
                if computer_name_for_file == computer_name:
                    try:
                        data_list.append(pd.read_csv(self.terrarium_folder_path+'/'+self.terrarium_log_file_list[i],usecols=self.used_column_list))
                    except:
                        print("Has to use input/output columns")
                        try:
                            data_list.append(pd.read_csv(self.terrarium_folder_path+'/'+self.terrarium_log_file_list[i],usecols=self.used_column_list_with_inputs))
                        except Exception as e:
                            exit(e)
            computer_data = pd.concat(data_list).reset_index(drop=True)
            for column in computer_data.columns:
                if not column.find('/inputs') == -1:
                    computer_data = computer_data.rename(columns={column:column[7:]})
            self.all_log_data.update({computer_name.upper():computer_data})


    def split_data_into_runs(self,computer_name):
        print("Starting run splitting for " + computer_name + " data:")
        start_history_df = self.start_history_events_list[computer_name]
        history_df = self.history_events_list[computer_name]
        log_data = self.all_log_data[computer_name]
        if start_history_df.equals(history_df):
            for i in range(len(start_history_df)):
                run_id = int(start_history_df['run_number'].iloc[i])
                start_closest_index = int((log_data['/timestamp']-start_history_df['timestamp'].iloc[i]).abs().argsort()[:1])
                if not i == len(start_history_df)-1:
                    stop_closest_index = int((log_data['/timestamp']-start_history_df['timestamp'].iloc[i+1]).abs().argsort()[:1])
                else:
                    stop_closest_index = len(log_data)
                run = log_data.iloc[start_closest_index:stop_closest_index,:]
                if len(run) != 0 and stop_closest_index != len(log_data):
                    filename = self.split_terrarium_data_folder_path + "/"+ computer_name + "_run_{}.csv".format(str(run_id).zfill(4))
                    run.to_csv(filename,index=False)
        else:
            for i in range(len(start_history_df)):
                run_id = int(start_history_df['run_number'].iloc[i])
                start_closest_index = int((log_data['/timestamp']-start_history_df['timestamp'].iloc[i]).abs().argsort()[:1])
                stop_timestamp = float(history_df[(history_df['run_number']==run_id) & (history_df['type']=='RUN_COMPLETE')]['timestamp'])
                stop_closest_index = int((log_data['/timestamp']-stop_timestamp).abs().argsort()[:1])
                run = log_data.iloc[start_closest_index:stop_closest_index,:]
                if run_id == 1 and self.apply_operator_offset:
                    self.height_offset = run['/brain_outputs/robot_position_ft'].iloc[0]
                if len(run) != 0 and stop_closest_index != len(log_data):
                    filename = self.split_terrarium_data_folder_path + "/"+ computer_name + "_run_{}.csv".format(str(run_id).zfill(4))
                    run.to_csv(filename,index=False) 
        print("Split and saved " + computer_name + " data")  

    def convert_to_utscope(self,bad_files):
        Q = np.identity(2)*self.Q_set
        Rm = np.identity(2)*self.Rm_set
        Rwb = np.identity(2)*self.Rwb_set
        g_scale = self.gscale_set
        self.raw_run_file_list = sorted( filter( lambda x: os.path.isfile(os.path.join(self.split_terrarium_data_folder_path, x)),
                        os.listdir(self.split_terrarium_data_folder_path) ) )        
        for i in range(0,len(self.raw_run_file_list)):
            id_end = self.raw_run_file_list[i].rfind('.')
            id_start = self.raw_run_file_list[i].rfind('_')
            run_id = int(self.raw_run_file_list[i][id_start+1:id_end].replace('0',''))
            if run_id not in bad_files:
                self.convert_run_to_utscope(self.raw_run_file_list[i],Q,Rm,Rwb,g_scale)
        self.compress_uts_data_into_zip()
        print(f"All runs converted to utscope format for slug {self.slug_name}")

    def convert_run_to_utscope(self,run_name,Q,Rm,Rwb,g_scale):
        id_location = run_name.rfind('_')
        computer_index = run_name.find('_')
        run_id = run_name[id_location+1:-4]
        computer_name = run_name[:computer_index]
        run_file_location = self.split_terrarium_data_folder_path + '/' + run_name
        run = odm.runData(run_file_location,self.vel_offset,self.GT_offset,'Terrarium','ft',self.height_offset)
        run.applyKVC(Q,Rm,Rwb,g_scale,1)
        ut_scope_data = run.Kvc[['time','s']]
        ut_scope_data = ut_scope_data.rename(columns={'time':'time (s)','s':'encoder displacement (ft)'})
        ut_scope_data_file_location = self.utscope_data_path + '/' + computer_name + '_run' + run_id + '_localization.csv'
        save_df(ut_scope_data,ut_scope_data_file_location)
        ut_scope_graph_file_location = self.utscope_plots_path + '/' + computer_name + '_run' + run_id + '_localization.png'
        run.save_total_distance_plot(ut_scope_graph_file_location)

    def compress_uts_data_into_zip(self):
        zip_file_name = self.directory + '/' + self.slug_name + '.zip'
        if os.path.exists(zip_file_name):
            os.remove(self.directory+'/'+self.slug_name+'.zip')
        shutil.make_archive(self.directory+'/'+self.slug_name, 'zip', self.utscope_data_path)
        
def save_df(df,file_location):
    df.to_csv(file_location,index=False)

def convert_timestamp_to_date(timestamp):
    # convert time stamp in '2023-04-28T19:17:29.378639+00:00' to a date "04-28-2023"
    date = timestamp[:10]
    return date

if __name__ == "__main__":
    slug_list = {
                # "20230131-7e2700":False,
                # "20230615-086467":True,
                "20230508-04487e":True,
                # "20230516-3c4686":True,
                # "20230427-e92647":False,
                # "20230522-1ac0e6":False,
                }
    for slug_name in slug_list:
        print("Starting conversion for slug " + slug_name + "...")
        slug = SlugData(slug_name, "/home/tyler.harp/UTScopeData",is_data_already_there=True,apply_operator_offset=slug_list[slug_name])
        slug.break_up_into_runs()
        slug.convert_to_utscope([])


