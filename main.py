# import numpy as np
# import pandas as pd
# import matplotlib as plt
# import cv2
# from tabulate import tabulate
#
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
#
# import pytesseract
# from pytesseract import Output
#
# from ultralyticsplus import YOLO, render_result
# from PIL import Image
#
# image = 'page0.png'
#
# img = Image.open(image)
#
# # load model
# model = YOLO('keremberke/yolov8m-table-extraction')
#
# # set model parameters
# model.overrides['conf'] = 0.25  # NMS confidence threshold
# model.overrides['iou'] = 0.45  # NMS IoU threshold
# model.overrides['agnostic_nms'] = True  # NMS class-agnostic
# model.overrides['max_det'] = 1000  # maximum number of detections per image
#
# # perform inference
# results = model.predict(img)
# # results.show()
#
# # observe results
# # print('Boxes: ', results[0].boxes)
# render = render_result(model=model, image=img, result=results[0])
# render.show()
#
# x1, y1, x2, y2, _, _ = tuple(int(item) for item in results[0].boxes.data.numpy()[2])
# img = np.array(Image.open(image))
#
# cropped_image = img[y1:y2, x1:x2]
# cropped_image = Image.fromarray(cropped_image)
# # cv2.imshow("Image", cropped_image)
# # text = pytesseract.image_to_string(cropped_image)
# # print(text)
#
# ext_df = pytesseract.image_to_string(cropped_image)
# # print(ext_df)
#
# # print(tabulate(ext_df, headers="keys", tablefmt="psql"))
# print(ext_df)


"""Cell recognition algorithm."""
# import numpy as np
# import cv2
# from imutils import resize
#
# # demo document
# dem = dict(
#     img_path='page0Table.png',
#     gt_boxes=np.array([[451, 67, 749, 749]]),
#     in_boxes=np.array([[455, 84, 760, 785]]))
#
# dem_wdth = dem['in_boxes'][0][3] - dem['in_boxes'][0][1]
# dem_hght = dem['in_boxes'][0][2] - dem['in_boxes'][0][0]
# dem_xmin = dem['in_boxes'][0][1]
# dem_ymin = dem['in_boxes'][0][0]
#
# dem_image = cv2.imread(dem['img_path'])
#
# # detected table from document
# # tbl_image = dem_image[dem_ymin: dem_ymin + dem_hght,
# #                       dem_xmin: dem_xmin + dem_wdth]
#
# # threshold and resize table image
# tbl_gray = cv2.cvtColor(dem_image, cv2.COLOR_BGR2GRAY)
# tbl_thresh_bin = cv2.threshold(tbl_gray, 127, 255, cv2.THRESH_BINARY)[1]
#
# R = 2.5
# tbl_resized = resize(tbl_thresh_bin, width=int(dem_image.shape[1] // R))
#
#
# def get_dividers(img, axis):
#     """Return array indicies of white horizontal or vertical lines."""
#     blank_lines = np.where(np.all(img == 255, axis=axis))[0]
#     filtered_idx = np.where(np.diff(blank_lines) != 1)[0]
#     return blank_lines[filtered_idx]
#
#
# dims = dem_image.shape[0], dem_image.shape[1]
#
# # table mask to search for gridlines
# tbl_str = np.zeros(dims, np.uint8)
# tbl_str = cv2.rectangle(tbl_str, (0, 0), (dims[1] - 1, dims[0] - 1), 255, 1)
#
# for a in [0, 1]:
#     dividers = get_dividers(tbl_resized, a)
#     start_point = [0, 0]
#     end_point = [dims[1], dims[1]]
#     for i in dividers:
#         i *= R
#         start_point[a] = int(i)
#         end_point[a] = int(i)
#         cv2.line(tbl_str,
#                  tuple(start_point),
#                  tuple(end_point),
#                  255,
#                  1)
#
# contours, hierarchy = cv2.findContours(tbl_str,
#                                        cv2.RETR_TREE,
#                                        cv2.CHAIN_APPROX_SIMPLE)
#
#
# def sort_contours(cnts, method="left-to-right"):
#     """Return sorted countours."""
#     reverse = False
#     k = 0
#     if method in ['right-to-left', 'bottom-to-top']:
#         reverse = True
#     if method in ['top-to-bottom', 'bottom-to-top']:
#         k = 1
#     b_boxes = [cv2.boundingRect(c) for c in cnts]
#     (cnts, b_boxes) = zip(*sorted(zip(cnts, b_boxes),
#                                   key=lambda b: b[1][k],
#                                   reverse=reverse))
#     return (cnts, b_boxes)
#
#
# contours, boundingBoxes = sort_contours(contours, method='top-to-bottom')
#
# # remove countours of the whole table
# bb_filtered = [list(t) for t in boundingBoxes
#                if t[2] < dims[1] and t[3] < dims[0]]
#
# # allocate countours in table-like structure
# rows = []
# columns = []
#
# for i, bb in enumerate(bb_filtered):
#     if i == 0:
#         columns.append(bb)
#         previous = bb
#     else:
#         if bb[1] < previous[1] + previous[3] / 2:
#             columns.append(bb)
#             previous = bb
#             if i == len(bb_filtered) - 1:
#                 rows.append(columns)
#         else:
#             rows.append(columns)
#             columns = []
#             previous = bb
#             columns.append(bb)


data = [{'row': [-3.7727012634277344,
   0.06823517382144928,
   1214.89794921875,
   24.892118453979492],
  'cells': [{'column': [7.207976818084717,
     0.24542659521102905,
     84.62327575683594,
     113.63933563232422],
    'cell': [7.207976818084717,
     0.06823517382144928,
     84.62327575683594,
     24.892118453979492]},
   {'column': [82.82319641113281,
     0.22280320525169373,
     442.02728271484375,
     113.40961456298828],
    'cell': [82.82319641113281,
     0.06823517382144928,
     442.02728271484375,
     24.892118453979492]},
   {'column': [437.5136413574219,
     0.19758880138397217,
     649.78271484375,
     113.51570892333984],
    'cell': [437.5136413574219,
     0.06823517382144928,
     649.78271484375,
     24.892118453979492]},
   {'column': [650.1903076171875,
     0.20577654242515564,
     772.2666625976562,
     113.4121322631836],
    'cell': [650.1903076171875,
     0.06823517382144928,
     772.2666625976562,
     24.892118453979492]},
   {'column': [769.6044311523438,
     0.23378074169158936,
     919.8126220703125,
     113.42961883544922],
    'cell': [769.6044311523438,
     0.06823517382144928,
     919.8126220703125,
     24.892118453979492]},
   {'column': [918.8020629882812,
     0.1754520833492279,
     1077.8978271484375,
     113.43669891357422],
    'cell': [918.8020629882812,
     0.06823517382144928,
     1077.8978271484375,
     24.892118453979492]},
   {'column': [1076.200927734375,
     0.28093308210372925,
     1215.140869140625,
     113.6121597290039],
    'cell': [1076.200927734375,
     0.06823517382144928,
     1215.140869140625,
     24.892118453979492]}],
  'cell_count': 7},
 {'row': [-3.5536317825317383,
   24.60768699645996,
   1214.97119140625,
   54.56975555419922],
  'cells': [{'column': [7.207976818084717,
     0.24542659521102905,
     84.62327575683594,
     113.63933563232422],
    'cell': [7.207976818084717,
     24.60768699645996,
     84.62327575683594,
     54.56975555419922]},
   {'column': [82.82319641113281,
     0.22280320525169373,
     442.02728271484375,
     113.40961456298828],
    'cell': [82.82319641113281,
     24.60768699645996,
     442.02728271484375,
     54.56975555419922]},
   {'column': [437.5136413574219,
     0.19758880138397217,
     649.78271484375,
     113.51570892333984],
    'cell': [437.5136413574219,
     24.60768699645996,
     649.78271484375,
     54.56975555419922]},
   {'column': [650.1903076171875,
     0.20577654242515564,
     772.2666625976562,
     113.4121322631836],
    'cell': [650.1903076171875,
     24.60768699645996,
     772.2666625976562,
     54.56975555419922]},
   {'column': [769.6044311523438,
     0.23378074169158936,
     919.8126220703125,
     113.42961883544922],
    'cell': [769.6044311523438,
     24.60768699645996,
     919.8126220703125,
     54.56975555419922]},
   {'column': [918.8020629882812,
     0.1754520833492279,
     1077.8978271484375,
     113.43669891357422],
    'cell': [918.8020629882812,
     24.60768699645996,
     1077.8978271484375,
     54.56975555419922]},
   {'column': [1076.200927734375,
     0.28093308210372925,
     1215.140869140625,
     113.6121597290039],
    'cell': [1076.200927734375,
     24.60768699645996,
     1215.140869140625,
     54.56975555419922]}],
  'cell_count': 7},
 {'row': [-3.386270046234131,
   56.99028778076172,
   1215.1427001953125,
   87.25006103515625],
  'cells': [{'column': [7.207976818084717,
     0.24542659521102905,
     84.62327575683594,
     113.63933563232422],
    'cell': [7.207976818084717,
     56.99028778076172,
     84.62327575683594,
     87.25006103515625]},
   {'column': [82.82319641113281,
     0.22280320525169373,
     442.02728271484375,
     113.40961456298828],
    'cell': [82.82319641113281,
     56.99028778076172,
     442.02728271484375,
     87.25006103515625]},
   {'column': [437.5136413574219,
     0.19758880138397217,
     649.78271484375,
     113.51570892333984],
    'cell': [437.5136413574219,
     56.99028778076172,
     649.78271484375,
     87.25006103515625]},
   {'column': [650.1903076171875,
     0.20577654242515564,
     772.2666625976562,
     113.4121322631836],
    'cell': [650.1903076171875,
     56.99028778076172,
     772.2666625976562,
     87.25006103515625]},
   {'column': [769.6044311523438,
     0.23378074169158936,
     919.8126220703125,
     113.42961883544922],
    'cell': [769.6044311523438,
     56.99028778076172,
     919.8126220703125,
     87.25006103515625]},
   {'column': [918.8020629882812,
     0.1754520833492279,
     1077.8978271484375,
     113.43669891357422],
    'cell': [918.8020629882812,
     56.99028778076172,
     1077.8978271484375,
     87.25006103515625]},
   {'column': [1076.200927734375,
     0.28093308210372925,
     1215.140869140625,
     113.6121597290039],
    'cell': [1076.200927734375,
     56.99028778076172,
     1215.140869140625,
     87.25006103515625]}],
  'cell_count': 7},
 {'row': [-3.550010919570923,
   88.61997985839844,
   1215.1201171875,
   113.72361755371094],
  'cells': [{'column': [7.207976818084717,
     0.24542659521102905,
     84.62327575683594,
     113.63933563232422],
    'cell': [7.207976818084717,
     88.61997985839844,
     84.62327575683594,
     113.72361755371094]},
   {'column': [82.82319641113281,
     0.22280320525169373,
     442.02728271484375,
     113.40961456298828],
    'cell': [82.82319641113281,
     88.61997985839844,
     442.02728271484375,
     113.72361755371094]},
   {'column': [437.5136413574219,
     0.19758880138397217,
     649.78271484375,
     113.51570892333984],
    'cell': [437.5136413574219,
     88.61997985839844,
     649.78271484375,
     113.72361755371094]},
   {'column': [650.1903076171875,
     0.20577654242515564,
     772.2666625976562,
     113.4121322631836],
    'cell': [650.1903076171875,
     88.61997985839844,
     772.2666625976562,
     113.72361755371094]},
   {'column': [769.6044311523438,
     0.23378074169158936,
     919.8126220703125,
     113.42961883544922],
    'cell': [769.6044311523438,
     88.61997985839844,
     919.8126220703125,
     113.72361755371094]},
   {'column': [918.8020629882812,
     0.1754520833492279,
     1077.8978271484375,
     113.43669891357422],
    'cell': [918.8020629882812,
     88.61997985839844,
     1077.8978271484375,
     113.72361755371094]},
   {'column': [1076.200927734375,
     0.28093308210372925,
     1215.140869140625,
     113.6121597290039],
    'cell': [1076.200927734375,
     88.61997985839844,
     1215.140869140625,
     113.72361755371094]}],
  'cell_count': 7}]

# print(data[0]["cells"])

data[0]["cells"][0]["column"] = [x+1 for x in data[0]["cells"][0]["column"]]

print(data)