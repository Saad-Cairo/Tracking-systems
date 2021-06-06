import numpy as np
import matplotlib.pyplot as plt
import csv
import copy
from scipy.optimize import curve_fit
import pickle
import os
import logging


def fit_func(x, a, b):
    return a * (x ** 2) + b


# get part of the track to fit a curve on it

def get_subtrack(track, cut_dir, delta_time=20):
    """
    track = {'start': ,    'end': ,    'points_x': [],    'points_y': []}
    delta_time: time to be cut on
    cut_dir: 1 to cut from the end 0 to cut from the start
    """

    subtrack = {'start': [],
                'end': [],
                'points': [],
                'params': [],
                'idx': 0
                }
    if delta_time > len(track['points']):
        delta_time = len(track['points'])

    if cut_dir == 0:
        subtrack['points'] = track['points'][0:delta_time]
        subtrack['start'] = subtrack['points'][0]
        subtrack['end'] = subtrack['points'][-1]
        subtrack['idx'] = track['idx']
    else:
        subtrack['points'] = track['points'][-delta_time:]
        subtrack['start'] = track['points'][-delta_time]
        subtrack['end'] = track['points'][-1]
        subtrack['idx'] = track['idx']

    return subtrack


# fit a curve
def fit_curve(track):
    params = []
    x_fit = []

    y_fit = []
    points = np.asarray(track['points'])
    points_x = points[:, 0]
    points_y = points[:, 1]

    params = curve_fit(fit_func, points_x, points_y)[0]
    # 100 linearly spaced numbers
    x_fit = np.linspace(points_x[0], points_x[len(points_x) - 1], 100)

    # the function, which is y_fit = a*(x_fit**2)+b
    y_fit = params[0] * (x_fit ** 2) + params[1]

    plt.plot(x_fit, y_fit, 'r')
    track['params'] = params
    return track


# predict points from curve
def get_dist_from_prediction_of_tracks(sub_track1, sub_track2):
    """
    params1 : [a,b] for the first sub_track "predict after the end"
    params2 : [a,b] for the second sub_track "predict before the start"
    sub_track1/sub_track2 : dictionaries of the 2 tracks
    """
    params1 = sub_track1['params']
    params2 = sub_track2['params']
    end_point_sub_track1 = sub_track1['points'][len(sub_track1['points']) - 1]
    start_point_sub_track2 = sub_track2['points'][0]
    middle_point_x = (end_point_sub_track1[0] + start_point_sub_track2[0]) / 2
    y1 = params1[0] * (middle_point_x ** 2) + params1[1]
    y2 = params2[0] * (middle_point_x ** 2) + params2[1]
    return np.abs(y1 - y2)


def get_difference_between_slopes_of_tracks(sub_track1, sub_track2):
    params1 = sub_track1['params']
    params2 = sub_track2['params']
    end_point_sub_track1 = sub_track1['points'][len(sub_track1['points']) - 1]
    start_point_sub_track2 = sub_track2['points'][0]
    slope1 = params1[0] * 2 * end_point_sub_track1[0]
    slope2 = params2[0] * 2 * start_point_sub_track2[0]
    diff = np.abs(slope1 - slope2)
    return diff


def get_difference_between_speeds_of_tracks(sub_track1, sub_track2):
    params1 = sub_track1['params']
    params2 = sub_track2['params']
    end_3_points_sub_track1 = sub_track1['points'][len(sub_track1['points']) - 3:len(sub_track1['points'])]
    start_3_point_sub_track2 = sub_track2['points'][0:3]
    point1_1 = end_3_points_sub_track1[0]
    point1_2 = end_3_points_sub_track1[1]
    point1_3 = end_3_points_sub_track1[2]
    point2_1 = start_3_point_sub_track2[0]
    point2_2 = start_3_point_sub_track2[1]
    point2_3 = start_3_point_sub_track2[2]

    speed1_12 = ((point1_1[0] - point1_2[0]) ** 2) + ((point1_1[1] - point1_2[1]) ** 2)
    speed1_23 = ((point1_2[0] - point1_3[0]) ** 2) + ((point1_2[1] - point1_3[1]) ** 2)
    speed1 = (speed1_12 + speed1_23) / 2

    speed2_12 = ((point2_1[0] - point2_2[0]) ** 2) + ((point2_1[1] - point2_2[1]) ** 2)
    speed2_23 = ((point2_2[0] - point2_3[0]) ** 2) + ((point2_2[1] - point2_3[1]) ** 2)
    speed2 = (speed2_12 + speed2_23) / 2

    diff = np.abs(speed1 - speed2)
    return diff


def slopes_metric(tracks_arr_1, tracks_arr_2):
    metric = np.ones((len(tracks_arr_1), len(tracks_arr_2))) * np.inf
    for i in range(len(tracks_arr_1)):
        for j in range(len(tracks_arr_2)):
            if tracks_arr_1[i]['idx'] != tracks_arr_2[j]['idx']:
                metric[i][j] = get_difference_between_slopes_of_tracks(tracks_arr_1[i], tracks_arr_2[j])

    return metric


def predict_points_metric(tracks_arr_1, tracks_arr_2):
    metric = np.ones((len(tracks_arr_1), len(tracks_arr_2))) * np.inf
    for i in range(len(tracks_arr_1)):
        for j in range(len(tracks_arr_2)):
            if tracks_arr_1[i]['idx'] != tracks_arr_2[j]['idx']:
                metric[i][j] = get_dist_from_prediction_of_tracks(tracks_arr_1[i], tracks_arr_2[j])

    return metric


def speed_metric(tracks_arr_1, tracks_arr_2):
    metric = np.ones((len(tracks_arr_1), len(tracks_arr_2))) * np.inf
    for i in range(len(tracks_arr_1)):
        for j in range(len(tracks_arr_2)):
            if tracks_arr_1[i]['idx'] != tracks_arr_2[j]['idx']:
                metric[i][j] = get_difference_between_speeds_of_tracks(tracks_arr_1[i], tracks_arr_2[j])

    return metric


def get_pairs_of_tracks(mat):
    '''sort the matrix by index'''
    B = [(i, j) for i, x in enumerate(mat) for j, _ in enumerate(x)]
    B.sort(key=lambda ix: mat[ix[0]][ix[1]])

    '''row and column index'''
    rows = [False for i, x in enumerate(mat)]
    columns = [False for i, x in enumerate(mat[0])]

    '''get the tracks'''
    tracks = []

    counter = 0
    while (len(tracks) != np.minimum(len(mat[0]), len(mat))):
        index = B[counter]
        if (not rows[index[0]] and not columns[index[1]]):
            rows[index[0]] = True
            columns[index[1]] = True
            tracks.append(index)
        counter = counter + 1

    return tracks


def get_newly_arrived_tracks_in_one_time_Stamp(times, time_stamp):
    new = []
    if time_stamp in times:
        if times[time_stamp][1] != []:
            return times[time_stamp][1]  # if the newly arrived tracks are already computed then just return them
        for track_data in times[time_stamp][0]:
            if track_data[1] == True:
                new.append(track_data[
                               0])  # append the track ids that first appeared in this timestamp if not already computed
    return new


def set_newly_arrived_tracks_in_all_time_stamps(times,
                                                num_of_considered_frames):  # each track can stitch to a all tracks that came after it in number of frames
    for t in reversed(sorted(list(times))):
        for i in range(num_of_considered_frames):
            cands_at_t = get_newly_arrived_tracks_in_one_time_Stamp(times, t + i)
            times[t][1] = times[t][1] + cands_at_t


def get_times_pos_of_a_track(path, t, times, track_id):
    xx = []
    yy = []
    with open(path, mode='r') as infile:
        rdr = csv.reader(infile)
        times_pos = {}
        p = 0
        new = False
        for row in rdr:
            if p == 1:
                new = True
            if p >= 1:
                x = (float(row[0]) + float(row[2])) / 2
                y = (float(row[1]) + float(row[3])) / 2
                xx.append(x)
                yy.append(y)
                times_pos[t] = [x, y]
                if t not in times:
                    tracklets_in_the_current_t = []
                    tracklets_in_the_current_t.append([track_id, new])
                    times[t] = [tracklets_in_the_current_t, []]
                else:
                    times[t][0].append([track_id, new])
                t += 1

            p += 1
            new = False
    return times_pos, times


def manipulate_data():
    # This is the path where all the files are stored.
    folder_path = 'Corrected_01_tracks'
    # Open one of the files,
    all_files = []
    track_ids = {}
    times = {}
    ts = []
    for data_file in (os.listdir(folder_path)):
        all_files.append(data_file)
    track_id = 1
    while (track_id < len(all_files) // 2):
        txt = 'Corrected_01_tracks/track_%d_info.txt' % track_id
        f = open(txt)
        start_time = int(f.readline())
        ts.append(start_time)
        csv_file = 'Corrected_01_tracks/track_%d.csv' % track_id
        track_ids[track_id], times = get_times_pos_of_a_track(csv_file, start_time, times, track_id)
        track_id += 1
    set_newly_arrived_tracks_in_all_time_stamps(times, 2)
    return track_ids


def get_center(x1, y1, x2, y2):
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def make_tracks(pickle_list):
    # objects = []
    # with (open(path_to_pickle_file, "rb")) as openfile:
    #     while True:
    #         try:
    #             objects.append(pickle.load(openfile))
    #         except EOFError:
    #             break

    objects = pickle_list

    x = objects[0]
    keys = x.keys()
    idx = 0
    tracks = []
    for i in x.keys():
        start_point = get_center(x[i][0][1], x[i][0][2], x[i][0][3], x[i][0][4])
        end_point = get_center(x[i][-1][1], x[i][-1][2], x[i][-1][3], x[i][-1][4])
        points = []
        for j in range(len(x[i])):
            p = get_center(x[i][j][1], x[i][j][2], x[i][j][3], x[i][j][4])
            points.append(p)

        start_frame_name = x[i][0][0]
        end_frame_name = x[i][-1][0]
        start_time = int(start_frame_name[2:-4])
        end_time = int(end_frame_name[2:-4])
        track = {
            'start': start_point,
            'end': end_point,
            'points': points,
            'start_time': start_time,
            'end_time': end_time,
            'idx': idx,
            'pickle_track': x[i]
        }

        if len(points) < 3:
            continue
        tracks.append(track)
        idx += 1
    return tracks


def split_on_time(min_time_window, tracks):
    windows = []
    length = len(tracks)
    i = 0
    starting_done = np.zeros((len(tracks)))
    ending_done = np.zeros((len(tracks)))
    while i < length:
        if starting_done[i] == 1:
            i += 1
            continue
        starting_done[i] = 1
        track = copy.deepcopy(tracks[i])
        window = {
            "starting_tracks": [],
            "ending_tracks": []
        }
        window["starting_tracks"].append(track)

        j = 0
        length = len(tracks)
        while j < length:
            neighbour_track = copy.deepcopy(tracks[j])

            if neighbour_track['idx'] == track['idx'] or ending_done[j] == 1:
                j += 1
                continue

            if (np.abs(track["start_time"] - neighbour_track["end_time"]) < min_time_window):
                ending_done[j] = 1
                window["ending_tracks"].append(neighbour_track)

            j += 1

        j = i + 1
        while j < length:
            neighbour_track = copy.deepcopy(tracks[j])
            if (np.abs(track["start_time"] - neighbour_track["start_time"]) < min_time_window):
                if starting_done[j] == 1 or ending_done[j] == 1:
                    j += 1
                    continue
                starting_done[j] = 1
                window["starting_tracks"].append(neighbour_track)

            j += 1
        print(window)
        windows.append(window)
        i += 1
    return windows


def euc_dist(point1, point2):
    return np.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2))


def split_on_pos(min_dist_window, time_windows):
    windows = []
    for w in range(len(time_windows)):
        time_window = time_windows[w]
        length = len(time_window["starting_tracks"])
        i = 0
        while i < length:
            track = copy.deepcopy(time_window["starting_tracks"][i])
            del time_window["starting_tracks"][i]
            i -= 1
            length -= 1
            window = {
                "starting_tracks": [],
                "ending_tracks": []
            }

            window["starting_tracks"].append(track)
            j = i + 1
            while j < length:
                neighbour_track = copy.deepcopy(time_window["starting_tracks"][j])
                if (euc_dist(track["start"], neighbour_track["start"]) < min_dist_window):
                    window["starting_tracks"].append(neighbour_track)
                    del time_window["starting_tracks"][j]
                    j -= 1
                    length -= 1
                j += 1

            length2 = len(time_window["ending_tracks"])

            j = 0
            while j < length2:
                neighbour_track = copy.deepcopy(time_window["ending_tracks"][j])

                if (euc_dist(track["start"], neighbour_track["end"]) < min_dist_window):
                    window["ending_tracks"].append(neighbour_track)
                j += 1

            windows.append(window)
            i += 1
    return windows


def log_details_of_windows(windows, connection_indices, metric):
    logger = logging.getLogger(__name__)

    k = 0
    for i in range(len(windows)):
        if (len(windows[i]["ending_tracks"]) > 0 and len(windows[i]["starting_tracks"]) > 0):
            logger.debug("window " + str(i))
            logger.debug('\n')
            for j in range(len(windows[i]["starting_tracks"])):
                logger.debug("starting track # " + str(j) + " :")
                logger.debug('\n')
                logger.debug("end time " + str(windows[i]["starting_tracks"][j]["end_time"]))
                logger.debug('\n')
                logger.debug("start time " + str(windows[i]["starting_tracks"][j]["start_time"]))
                logger.debug('\n')
                logger.debug("start point " + str(windows[i]["starting_tracks"][j]["start"]))
                logger.debug('\n')
                logger.debug("end point " + str(windows[i]["starting_tracks"][j]["end"]))
                logger.debug('\n')
                logger.debug("idx " + str(windows[i]["starting_tracks"][j]["idx"]))
                logger.debug('\n')
            for j in range(len(windows[i]["ending_tracks"])):
                logger.debug("ending track # " + str(j) + " :")
                logger.debug('\n')
                logger.debug("end time " + str(windows[i]["ending_tracks"][j]["end_time"]))
                logger.debug('\n')
                logger.debug("start time " + str(windows[i]["ending_tracks"][j]["start_time"]))
                logger.debug('\n')
                logger.debug("start point " + str(windows[i]["ending_tracks"][j]["start"]))
                logger.debug('\n')
                logger.debug("end point " + str(windows[i]["ending_tracks"][j]["end"]))
                logger.debug('\n')
                logger.debug("idx " + str(windows[i]["ending_tracks"][j]["idx"]))
                logger.debug('\n')

            logger.debug('\n')
            logger.debug("connections: " + str(connection_indices[k]))
            logger.debug('\n')
            logger.debug('\n')

            k += 1

    logger.debug("num of fragments " + str(k))


def connect_tracks(windows, metric):
    connections = []
    all_connection_indices = []
    for i in range(len(windows)):
        if (len(windows[i]["ending_tracks"]) > 0 and len(windows[i]["starting_tracks"]) > 0):

            starting_subtracks = []
            for st in windows[i]["starting_tracks"]:
                starting_subtracks.append(fit_curve(get_subtrack(st, 0)))

            ending_subtracks = []
            for en in windows[i]["ending_tracks"]:
                ending_subtracks.append(fit_curve(get_subtrack(en, 1)))

            if metric == "speed":
                connection_indices = get_pairs_of_tracks(speed_metric(starting_subtracks, ending_subtracks))
            elif metric == "slope":
                connection_indices = get_pairs_of_tracks(slopes_metric(starting_subtracks, ending_subtracks))
            else:
                connection_indices = get_pairs_of_tracks(predict_points_metric(starting_subtracks, ending_subtracks))

            all_connection_indices.append(connection_indices)

            for connection in connection_indices:
                ending_track_idx = windows[i]['ending_tracks'][connection[1]]['idx']
                starting_track_idx = windows[i]['starting_tracks'][connection[0]]['idx']
                connections.append((ending_track_idx, starting_track_idx))

    # log the details
    log_details_of_windows(windows, all_connection_indices, metric)

    connections_refined = []
    taken = np.zeros((len(connections)))
    for i in range(len(connections)):
        if taken[i] == 1:
            continue
        l = list(connections[i])
        for j in range(i + 1, len(connections), 1):
            if l[-1] == connections[j][0]:
                l.append(connections[j][1])
                taken[j] = 1
        connections_refined.append(tuple(l))

    return connections_refined


def connections_to_tracks(connections, tracks):
    linked_tracks = dict()
    taken = np.zeros((len(tracks)))
    track_idx = 0
    for connection in connections:
        track = []
        for idx in connection:
            track = track + tracks[idx]['pickle_track']
            taken[idx] = 1
        # linked_tracks.append(track)
        linked_tracks['AD' + str(track_idx)] = track
        track_idx += 1

    for i in range(len(tracks)):
        if taken[i] == 0:
            # linked_tracks.append(tracks[i]['pickle_track'])
            linked_tracks['AD' + str(track_idx)] = tracks[i]['pickle_track']
            track_idx += 1

    return linked_tracks


def connect_tracks_from_pickle(pickle_list, typ, min_time_window=10, min_dist_window=30):
    logger = logging.getLogger(__name__)
    logger.info("defragmentation started...")

    tracks = make_tracks(pickle_list)
    logger.info("tracks prepared...")

    windows = split_on_time(min_time_window, tracks)
    windows = split_on_pos(min_dist_window, windows)
    logger.info("windows of fragmentation detected")

    if typ == 'speed':
        speed_connections = connect_tracks(windows, 'speed')
        speed_linked_tracks = connections_to_tracks(speed_connections, tracks)
        with open('speed_linked_tracks.pkl', 'wb') as handle:
            pickle.dump(speed_linked_tracks, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("connected due to speeds")
        return speed_linked_tracks

    if typ == 'slope':
        slope_connections = connect_tracks(windows, 'slope')
        slope_linked_tracks = connections_to_tracks(slope_connections, tracks)
        with open('slope_linked_tracks.pkl', 'wb') as handle:
            pickle.dump(slope_linked_tracks, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("connected due to slopes")
        return slope_linked_tracks

    if typ == 'point':
        point_connections = connect_tracks(windows, 'point')
        point_linked_tracks = connections_to_tracks(point_connections, tracks)
        with open('point_linked_tracks.pkl', 'wb') as handle:
            pickle.dump(point_linked_tracks, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("connected due to euclidean distance")
        return point_linked_tracks

    logger.info("defragmentation done...")

# test

# path_to_pickle_file = '23.pkl'
# objects = []
# with (open(path_to_pickle_file, "rb")) as openfile:
#     while True:
#         try:
#             objects.append(pickle.load(openfile))
#         except EOFError:
#             break

# new_tracks = connect_tracks_from_pickle(objects,'speed',min_time_window = 10,min_dist_window = 30)
