#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import imutils
from PIL import ImageGrab
from math import floor, ceil
from enum import Enum
import time
import win32api, win32con, time
import rich.status
from rich.progress import (
    TimeElapsedColumn,
    TextColumn
)

valid_bytes = ['1C', '7A', '55', 'BD', 'E9', 'FF']
header_template_o = cv2.imread('./templates/code_matrix_header.png')
header_template = cv2.cvtColor(header_template_o, cv2.COLOR_BGR2GRAY)


byte_templates = {}
for byte in valid_bytes:
    byte_template =  cv2.imread('./templates/' + byte.lower() + '.png')
    byte_templates[byte] = cv2.cvtColor(byte_template, cv2.COLOR_BGR2GRAY)
target_byte_templates = {}
for byte in valid_bytes:
    byte_template =  cv2.imread('./templates/t_' + byte.lower() + '.png')
    target_byte_templates[byte] = cv2.cvtColor(byte_template, cv2.COLOR_BGR2GRAY)


def find_matches(image_input, template, steps = 20):
    loc = False
    threshold = 0.9
    w, h = template.shape[::-1]
    # TODO: The code bellow can be used to work for varying resolution matching
    # for scale in np.append(np.linspace(0.2, 1.0, steps)[::-1], np.linspace(1.0, 2.0, steps)[1:]):
    #     resized = imutils.resize(template, width = int(template.shape[1] * scale))
    #     w, h = resized.shape[::-1]
    #     res = cv2.matchTemplate(image_input,resized,cv2.TM_CCOEFF_NORMED)

    #     loc = np.where( res >= threshold)
    #     if len(list(zip(*loc[::-1]))) > 0:
    #         # print(f'Matched on scale {scale}')
    #         break
    res = cv2.matchTemplate(image_input,template,cv2.TM_CCOEFF_NORMED)
    loc = np.where( res >= threshold)

    matches = []
    mask = np.zeros(image_input.shape[:2], np.uint8)
    for pt in zip(*loc[::-1]):
        if mask[pt[1] + int(round(h/2)), pt[0] + int(round(w/2))] != 255:
            mask[pt[1]:pt[1]+h, pt[0]:pt[0]+w] = 255
            matches.append((pt[0],pt[1],pt[0]+w,pt[1]+h))
    return matches, (w,h), 1 #scale


status = rich.status.Status("Scanning for Breach")
status.start()
while True:
    screenshot = ImageGrab.grab()
    image_o = np.array(screenshot)[:, :, ::-1]
    image = cv2.cvtColor(image_o, cv2.COLOR_BGR2GRAY)
    solution_start = time.process_time()
    matches, match_size, scale = find_matches(image, header_template)
    elapsed_time = time.process_time() - solution_start
    status.update(f"Scanning for Breach [{round(elapsed_time,2)}s]")
    if len(matches):
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -10000, -10000, 0, 0)
        time.sleep(1)
        status.stop()
        break
screenshot = ImageGrab.grab()
image_o = np.array(screenshot)[:, :, ::-1]
image = cv2.cvtColor(image_o, cv2.COLOR_BGR2GRAY)


matches, match_size, scale = find_matches(image, header_template)
header_pos = matches[0]


matched_bytes = {}
playing_field_crop_o = image_o[header_pos[1]:, header_pos[0]:]
playing_field_crop = cv2.cvtColor(playing_field_crop_o, cv2.COLOR_BGR2GRAY)
for byte, template in byte_templates.items():
    matches, match_size, scale = find_matches(playing_field_crop, template)
    matched_bytes[byte] = matches
matched_bytes_flat = [item for sublist in matched_bytes.values() for item in sublist]

space_cf = 0.05
min_x, max_x, min_y, max_y = min(matched_bytes_flat, key=lambda x: x[1])[1], max(matched_bytes_flat, key=lambda x: x[3])[3], min(matched_bytes_flat, key=lambda x: x[0])[0], max(matched_bytes_flat, key=lambda x: x[2])[2]
min_x = int(round(min_x - (max_x-min_x)*space_cf))
min_y = int(round(min_y - (max_y-min_y)*space_cf))
max_x = int(round(max_x + (max_x-min_x)*space_cf))
max_y = int(round(max_y + (max_y-min_y)*space_cf))
cropped_field = playing_field_crop_o[min_x:max_x, min_y:max_y]

matched_highlight_field = cv2.cvtColor(cv2.cvtColor(cropped_field, cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2RGB)
for byte, matches in matched_bytes.items():
    for match in matches:
        byte_template = byte_templates[byte]
        y1, y2 = match[0] - min_y, match[2] - min_y
        x1, x2 = match[1] - min_x, match[3] - min_x
        cv2.rectangle(matched_highlight_field, (y1-1, x1-1), (y2, x2), (0,255,0), 0)
        highlighted_byte = cv2.merge((byte_template, byte_template, byte_template))
        highlighted_byte[:,:,0] = 0
        highlighted_byte[:,:,1] = 0
        matched_highlight_field[x1:x2, y1:y2] = highlighted_byte

buffer_template_o = cv2.imread('./templates/buffer_frame.png')
buffer_template = cv2.cvtColor(buffer_template_o, cv2.COLOR_BGR2GRAY)
buffer_matches, match_size, scale = find_matches(image, buffer_template)
buffer_size = len(buffer_matches)
print(f'Buffer size - {buffer_size}')

buffer_min_x, buffer_max_x, buffer_min_y, buffer_max_y = min(buffer_matches, key=lambda x: x[1])[1], max(buffer_matches, key=lambda x: x[3])[3], min(buffer_matches, key=lambda x: x[0])[0], max(buffer_matches, key=lambda x: x[2])[2]
buffer_min_x = int(round(buffer_min_x - (buffer_max_x-buffer_min_x)*0.7))
buffer_min_y = int(round(buffer_min_y - (buffer_max_y-buffer_min_y)*0.08))
buffer_max_x = int(round(buffer_max_x + (buffer_max_x-buffer_min_x)*0.4))
buffer_max_y = int(round(buffer_max_y + (buffer_max_y-buffer_min_y)*0.08))
cropped_buffer = image_o[buffer_min_x:buffer_max_x, buffer_min_y:buffer_max_y]
matched_highlight_buffer = cv2.cvtColor(cv2.cvtColor(cropped_buffer, cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2RGB)
for match in buffer_matches:
    y1, y2 = match[0] - buffer_min_y, match[2] - buffer_min_y
    x1, x2 = match[1] - buffer_min_x, match[3] - buffer_min_x
    cv2.rectangle(matched_highlight_buffer, (y1-5, x1-5), (y2+5, x2+5), (0,255,0), 0)
    highlighted_buffer = cv2.merge((buffer_template, buffer_template, buffer_template))
    highlighted_buffer[:,:,0] = 0
    highlighted_buffer[:,:,1] = 0
    matched_highlight_buffer[x1:x2, y1:y2] = highlighted_buffer

def mean_unique(input_list, deviation_percentage = 0.05):
    input_list = input_list.copy()
    input_list.sort()
    average = input_list[0]
    minimal, maximum = average, average
    result = []
    current_values = []
    for input_item in input_list:
        if input_item < average - average*deviation_percentage or input_item > average + average*deviation_percentage:
            result.append({
                "average": average,
                "minimal": minimal,
                "maximum": maximum,
                "values": current_values
            })
            average = input_item
            minimal, maximum = average, average
            current_values = []
        else:
            current_values.append(input_item)
            average = sum(current_values) / len(current_values)
            minimal = min(minimal, input_item)
            maximum = max(maximum, input_item)
    result.append({
        "average": average,
        "minimal": minimal,
        "maximum": maximum,
        "values": current_values
    })
    return result


x_means = mean_unique([x[0] for x in matched_bytes_flat])
y_means = mean_unique([x[1] for x in matched_bytes_flat])

cells_positions = {}
rows = []
for y,y_mean in enumerate(y_means):
    row = []
    cells_positions[y] = {}
    for x, x_mean in enumerate(x_means):
        for byte, matches in matched_bytes.items():
            for match in matches:
                if match[0] >= x_mean['minimal'] and match[0] <= x_mean["maximum"] and match[1] >= y_mean['minimal'] and match[1] <= y_mean["maximum"]:
                    row.append(byte)    
                    cells_positions[y][x] = match
    print(row)
    rows.append(row)

matched_target_bytes = {}
cropped_non_field_o = playing_field_crop_o[:int(image.shape[0]-min_x-image.shape[0]*0.35), max_y:int(image.shape[1]-min_y-image.shape[1]*0.075)]
cropped_non_field = cv2.cvtColor(cropped_non_field_o, cv2.COLOR_BGR2GRAY)
for byte, template in target_byte_templates.items():
    matches, match_size, scale = find_matches(cropped_non_field, template)
    matched_target_bytes[byte] = matches
matched_target_bytes_flat = [item for sublist in matched_target_bytes.values() for item in sublist]
matched_highlight = cropped_non_field_o.copy()
for match in matched_target_bytes_flat:
    cv2.rectangle(matched_highlight, match[:2], match[2:], (0,0,255), 2)

space_cf = 0.005
targets_min_x, targets_max_x, targets_min_y, targets_max_y = min(matched_target_bytes_flat, key=lambda x: x[1])[1], max(matched_target_bytes_flat, key=lambda x: x[3])[3], min(matched_target_bytes_flat, key=lambda x: x[0])[0], cropped_non_field_o.shape[1]
targets_min_x = int(round(targets_min_x - (targets_max_x-targets_min_x)*(space_cf*15)))
targets_min_y = int(round(targets_min_y - (targets_max_y-targets_min_y)*space_cf))
targets_max_x = int(round(targets_max_x + (targets_max_x-targets_min_x)/len(matched_target_bytes_flat)*(1.3)))
targets_max_y = int(round(targets_max_y - (targets_max_y-targets_min_y)*(space_cf*2)))
cropped_non_field = cropped_non_field_o[targets_min_x:targets_max_x, targets_min_y:targets_max_y]
matched_highlight_targets = cropped_non_field.copy()
for target_byte, target_matches in matched_target_bytes.items():
    for target_match in target_matches:
        byte_template = target_byte_templates[target_byte]
        y1, y2 = target_match[0] - targets_min_y, target_match[2] - targets_min_y
        x1, x2 = target_match[1] - targets_min_x, target_match[3] - targets_min_x
        cv2.rectangle(matched_highlight_targets, (y1-1, x1-1), (y2, x2), (0,255,0), 0)
        highlighted_byte = cv2.merge((byte_template, byte_template, byte_template))
        highlighted_byte[:,:,0] = 0
        highlighted_byte[:,:,1] = 0
        matched_highlight_targets[x1:x2, y1:y2] = highlighted_byte

target_x_means = mean_unique([x[0] for x in matched_target_bytes_flat])
target_y_means = mean_unique([x[1] for x in matched_target_bytes_flat])

target_rows = []
for target_n, y_mean in enumerate(target_y_means):
    row = []
    for x_mean in target_x_means:
        for byte, matches in matched_target_bytes.items():
            for match in matches:
                if match[0] >= x_mean['minimal'] and match[0] <= x_mean["maximum"] and match[1] >= y_mean['minimal'] and match[1] <= y_mean["maximum"]:
                    row.append(byte)
    print(f"Target {target_n+1} - {row}")
    target_rows.append(row)

summary_image = np.zeros((matched_highlight_field.shape[0] + matched_highlight_targets.shape[0],matched_highlight_targets.shape[1],3), np.uint8)
summary_image[0:matched_highlight_targets.shape[0], 0:matched_highlight_targets.shape[1]] = matched_highlight_targets
summary_image[matched_highlight_targets.shape[0]:summary_image.shape[0], 0:matched_highlight_field.shape[1]] = matched_highlight_field
summary_image[matched_highlight_targets.shape[0]:matched_highlight_targets.shape[0]+matched_highlight_buffer.shape[0],matched_highlight_field.shape[1]:matched_highlight_field.shape[1]+matched_highlight_buffer.shape[1]] = matched_highlight_buffer
text_start_y, text_start_x = matched_highlight_targets.shape[0] + matched_highlight_buffer.shape[0] + 30, matched_highlight_field.shape[1] + 5
cv2.putText(summary_image, f"Buffer size - {buffer_size}", (text_start_x, text_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
text_start_y += 20
cv2.putText(summary_image, f"Matrix:", (text_start_x, text_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
text_start_y += 40
for row in rows:
    for col in row:
        cv2.putText(summary_image, col, (text_start_x, text_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        text_start_x += 50
    text_start_x = matched_highlight_field.shape[1] + 5
    text_start_y += 25
text_start_y += 20
cv2.putText(summary_image, f"Targets:", (text_start_x, text_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
text_start_y += 40
for target in target_rows:
    cv2.putText(summary_image, ':'.join(target), (text_start_x, text_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
    text_start_y += 25
cv2.imshow('Summary', summary_image)
cv2.setWindowProperty('Summary', cv2.WND_PROP_TOPMOST, 1)
cv2.waitKey(1500)
cv2.destroyAllWindows()

class Direction(Enum):
    ROW = 1
    COL = 2


def rank_solution(buffer, targets):
    buffer_str = ''.join(buffer)
    targets = [''.join(target) for target in targets]
    score = 0
    for n_target, target in enumerate(targets):
        if target in buffer_str:
            score += len(targets)-n_target+1
    return score



## Brute force solver
def walk_puzzle(rows, targets, buffer_left = 8, position = (Direction.COL, 0), buffer = [], path=[], used=[]):
    # print(f"Walking {position}, {buffer_left} buffer left")
    n_rows = len(rows)
    n_cols = len(rows[0])
    paths = {}
    if position[0] == Direction.COL:
        for n in range(n_cols):
            coords = (position[1], n)
            if coords in used:
                continue
            byte = rows[position[1]][n]
            new_buffer = [*buffer, byte]
            new_path = [*path, (Direction.ROW, n)]
            if buffer_left == 1:
                paths[tuple(new_path)] = (rank_solution(new_buffer, targets), new_buffer)
            else:
                paths = {**paths,**walk_puzzle(rows, targets, buffer_left-1, (Direction.ROW, n), new_buffer, new_path, [*used, coords])}
    else:
        for n in range(n_rows):
            coords = (n, position[1])
            if coords in used:
                continue
            byte = rows[n][position[1]]
            new_buffer = [*buffer, byte]
            new_path = [*path, (Direction.COL, n)]
            if buffer_left == 1:
                paths[tuple(new_path)] = (rank_solution(new_buffer, targets), new_buffer)
            else:
                paths = {**paths, **walk_puzzle(rows, targets, buffer_left-1, (Direction.COL, n), new_buffer, new_path, [*used, coords])}
    return paths

## Ranked walker
def walk_puzzle(rows, targets, buffer_left = 8, position = (Direction.COL, 0), buffer = [], path=[], used=[], max_score = 3):
    # print(f"Walking {position}, {buffer_left} buffer left")
    n_rows = len(rows)
    n_cols = len(rows[0])
    paths = {}
    if position[0] == Direction.COL:
        for n in range(n_cols):
            coords = (position[1], n)
            if coords in used:
                continue
            byte = rows[position[1]][n]
            new_buffer = [*buffer, byte]
            new_path = [*path, (Direction.ROW, n)]
            score = rank_solution(new_buffer, targets)
            paths[tuple(new_path)] = (score, new_buffer)
            if score >= max_score:
                return paths, tuple(new_path)
            if buffer_left != 1:
                walked, solution = walk_puzzle(rows, targets, buffer_left-1, (Direction.ROW, n), new_buffer, new_path, [*used, coords], max_score)
                paths = {**paths,**walked}
                if solution != None:
                    return paths, solution
    else:
        for n in range(n_rows):
            coords = (n, position[1])
            if coords in used:
                continue
            byte = rows[n][position[1]]
            new_buffer = [*buffer, byte]
            new_path = [*path, (Direction.COL, n)]
            score = rank_solution(new_buffer, targets)
            paths[tuple(new_path)] = (score, new_buffer)
            if score >= max_score:
                return paths, tuple(new_path)
            if buffer_left != 1:
                walked, solution = walk_puzzle(rows, targets, buffer_left-1, (Direction.COL, n), new_buffer, new_path, [*used, coords], max_score)
                paths = {**paths, **walked}
                if solution != None:
                    return paths, solution
    return paths, None

solving_status = rich.status.Status("Solving")
solving_status.start()
solution_start = time.process_time()
solutions, solution = walk_puzzle(rows, target_rows, buffer_size, max_score=sum([len(target_rows)-n+1 for n,r in enumerate(target_rows)]))
solving_status.stop()
print(f"Solved in {time.process_time() - solution_start}s")
if solution == None:
    solution = max(solutions, key=solutions.get)

solution_highlight = cropped_field.copy()
solution_highlight_img = image_o.copy()
current_row, current_col = 0, 0
positions = []
for step_n, step in enumerate(solution):
    if step[0] == Direction.ROW:
        cell_position = (current_col,step[1])
        position = cells_positions[current_col][step[1]]
        current_row = step[1]
    else:
        cell_position = (step[1], current_row)
        position = cells_positions[step[1]][current_row]
        current_col = step[1]
    cv2.rectangle(solution_highlight, (position[0] - min_y, position[1] - min_x), (position[2] - min_y, position[3] - min_x), (0,0,255), 2)
    cv2.putText(solution_highlight, f'{step_n+1}', (position[0]- min_y, position[1] - min_x), cv2.FONT_HERSHEY_SIMPLEX, 1, (25,255,255), 3)
    center_coord = (int((position[0]+position[2])/2+header_pos[0]), int((position[1]+position[3])/2+header_pos[1]))
    positions.append((center_coord, cell_position))
    cv2.circle(solution_highlight_img, center_coord, 20, (0,0,255), 2)
    text_size, _ = cv2.getTextSize(f'{step_n+1}', cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
    text_w, text_h = text_size
    cv2.putText(solution_highlight_img, f'{step_n+1}', (center_coord[0]-int(text_w/2), center_coord[1]-int(text_h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,100,255), 3)
cv2.imshow('Solution', solution_highlight)
cv2.setWindowProperty('Solution', cv2.WND_PROP_TOPMOST, 1)
cv2.waitKey(1000)
cv2.destroyAllWindows()
time.sleep(0.5)
for n_pos, (pos,cell_pos) in enumerate(positions):
    print(f"{n_pos} - {rows[cell_pos[0]][cell_pos[1]]} - {cell_pos} {pos}")
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -10000, -10000, 0, 0)
    time.sleep(0.01)
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, pos[0], pos[1], 0, 0)
    time.sleep(0.01)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
    time.sleep(0.01)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
    time.sleep(0.01)