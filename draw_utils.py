import cv2

def draw_boxes(frame, frame_number, x1_list, y1_list, x2_list, y2_list, id_list): 
    assert len(x1_list)==len(y1_list)==len(x2_list)==len(y2_list)==len(id_list)
    num_boxes = len(x1_list)
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    
    for ii in range(num_boxes): 
        x1, y1, x2, y2, person_id=(x1_list[ii], 
                                   y1_list[ii], 
                                   x2_list[ii], 
                                   y2_list[ii], 
                                   id_list[ii])
        # draw boxes on frame
        color = tuple([int((p * (person_id ** 2 - person_id + 1)) % 255
                          ) for p in palette])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, 
                    'id'+str(person_id), #+',confidence:'+str(conf),
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, # font scale 
                    color, 
                    3) # thickness
    # display crowd count at bottom left       
    cv2.putText(frame, "Crowd count: {}, Frame {}".format(num_boxes, frame_number),
               (10,frame.shape[0]-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, # font scale 
                (0,0,255), 
                3) # thickness
    return frame