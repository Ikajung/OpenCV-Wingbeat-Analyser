##############
# Techawin Manoonporn, KrappLab
# 1/6/2021
##############

import cv2 # openCV version 4.5.2
import numpy as np
import os
import time
import math
import matplotlib.colors as mcolors


# def get_filename(folder='.\\', filetype = '.mp4'):
# 	datalist = [f for f in os.listdir(folder) if filetype in f]
# 	for i in range(len(datalist)):
# 		print(i,' -> ',datalist[i])

# 	dataindex = int(input('[WAIT] Which data would you like to process? Choose number: '))
# 	if dataindex=='':
# 		print('[QUIT] no data was chosen.')
# 		exit()
# 	print('[INFO] #',dataindex,'-',datalist[dataindex],'- has been selected...')

# 	# para = datalist[dataindex][datalist[dataindex].find('[L')+1:datalist[dataindex].find('].mat')]
	
# 	# data = sio.loadmat(os.path.join(folder,datalist[dataindex]))['data']
# 	# data = numpy.transpose(data)
# 	return os.path.join(folder,datalist[dataindex])

def click_event(event, x, y, flags, params):
  
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
  
        # displaying the coordinates on the Shell
        print(x, ' ', y)
  
        # displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(demo, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('demo', demo)



####################################################################################################################

if __name__ == '__main__':

	videofile = "2021-06-08_142856_lucilia2_f20_r1.avi"
	# foldername = '.\\'
	# videofilename= get_filename(folder=foldername)
	# # print(videofilename)
	# # exit()
	
	# # videofilename = "C:\\Users\\huangjiaqi\\Desktop\\[Video][2019-04-03_22-30-08][L4V4].avi"
	# logfilename = '[Trajectory]'+videofilename.split('\\')[-1].split('.')[0][7::]+'['+foldername.split('\\')[-1]+']'
	# # print(logfilename)
	# # exit()
	# log_flag = 0
	
	# if log_flag==1:
	# 	log = open('.\\'+logfilename+'.csv','w+') 


	cap = cv2.VideoCapture(videofile)
	while not cap.isOpened():
		cap = cv2.VideoCapture(videofilename)
		cv2.waitKey(1000)
		print ("Wait for the header")


	pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) 


	w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
	h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float


	# video_output = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('x', 'v', 'i', 'd'), 30, (int(w),int(h)),True)	


	# flag, frame = cap.read()
	# prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# colours
	r = (0,0,255)
	g = (0,255,0)
	b = (255,0,0)
	c = (255,255,0)
	y = (0, 255, 255)
	m = (255, 0, 255)

	fn = open('behavioural_response_f20_r1_verticalax.csv', 'w')

	while True:
		flag, frame = cap.read()
		if flag:

			# convert from red color space to grayscale image
			gray = frame[::,::,2]

			# blurs the image, medianBlur is effective on removing salt-and-pepper noise
			gray = cv2.medianBlur(gray,5)

			# threshold to see wings, output is a binary image
			ret,thr = cv2.threshold(gray,25,255,cv2.THRESH_BINARY)	

			# erosion to remove noise in the case that SNR is low and requiring a low threshold
			# followed by dilation to increase the shrunken pixels
			kernel = np.ones((13,13),np.uint8) # controls amount of erosion/ dilation
			erosion = cv2.erode(thr, kernel, iterations = 1)
			# dilation = cv2.dilate(erosion, kernel, iterations = 1)

			thr = erosion

			# manually set the co-ordinates of the fixed points

			# centre of fly
			cx_mask = 648
			cy_mask= 473
			demo = frame.copy()
			cv2.circle(demo, (cx_mask,cy_mask), 5, y, 2)

			# wing hinges
			left_hinge = (cx_mask-33,455)
			cv2.circle(demo, left_hinge, 5, c, 2)

			right_hinge = (cx_mask+33,455)
			cv2.circle(demo, right_hinge, 5, c, 2)

			## applying masks // FRAME = (WIDTH, (-)HEIGHT) // (1280,720)

			black = 0 # black colour
			white = 255 # white colour
			thickness = -1 # -1 is to fill

			# create "big" white circle and black background (include area around the fly)
			mask_black = np.zeros(gray.shape[:2], dtype="uint8") # create black frame
			# c_big = (620,260) # centre point of big circle (x-axis, (-)y-axis)
			r_big = 250 # radius of big circle
			mask_circle_big = cv2.circle(mask_black, (cx_mask,cy_mask), r_big, white, thickness)

			# create small black circle mask (remove fly's body)

			mask_white1 = np.ones(gray.shape[:2], dtype="uint8")*255 # create white frame
			# c_small = (615,300) # centre point of small circle
			r_small = (90) # radius of small circle
			mask_circle_small = cv2.circle(mask_white1, (cx_mask,cy_mask), r_small, black, thickness)

			# create black rectangle mask (remove fly holder)

			mask_white2 = np.ones(gray.shape[:2], dtype="uint8")*255 # create white frame
			p1 = (cx_mask-55,0) # top left point of rectangle mask
			p2 = (cx_mask+50,720) # bottom right point of rectangle mask
			mask_rect = cv2.rectangle(mask_white2, p1, p2, black, thickness) 

			# create triangular mask (remove the legs)
			mask_white3 = np.ones(gray.shape[:2], dtype="uint8")*255 # create white frame
			t_top = (640,336)
			t_left = (518,720)
			t_right = (843,720)

			# cv2.circle(demo, t_top, 2, y, -1)
			# cv2.circle(demo, t_left, 2, y, -1)
			# cv2.circle(demo, t_right, 2, y, -1)

			triangle_cnt = np.array( [t_top, t_left, t_right] )
			mask_triangle = cv2.drawContours(mask_white3, [triangle_cnt], 0, black, -1)

			
			# combine all masks

			mask_comb = cv2.bitwise_and(mask_rect, mask_circle_big, mask = None)
			mask_comb = cv2.bitwise_and(mask_comb,mask_circle_small, mask = None)
			mask_comb = cv2.bitwise_and(mask_comb,mask_triangle, mask = None)

			roi_gray = cv2.bitwise_and(gray,gray,mask = mask_comb) # region of interest, on gray for calibration
			roi_thr = cv2.bitwise_and(thr,thr,mask = mask_comb) # region of interest, just the wings

			# roi_thr[600:720,0::]=0 # remove top 100 rows
####################################################################################################################
			contours,hierarchy = cv2.findContours(roi_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			# print("Number of Contours found = " + str(len(contours)))

			# all the contours found
			# cv2.drawContours(demo, contours, -1, b, 1)

			# initialise angles
			right_angle = 0
			left_angle = 0

			if len(contours) >= 2:
				for i in range(len(contours)):
					cnt = contours[i]
					area = cv2.contourArea(cnt)

					# print('area of cnt ' + str(i) + ' ' + str(area))

					# Extract only the contours of the wings using an area threshold
					# Convex Hull smoothens the contours
					if area > 7000 and area < 25000:
						hull = cv2.convexHull(cnt)
						# cv2.drawContours(demo, [hull], 0, c, 1)
					
						# centre of the contours
						M = cv2.moments(hull)
						cx = int(M['m10']/M['m00'])
						cy = int(M['m01']/M['m00'])

						# for left wing
						if cx <= cx_mask:
							# find the extreme points along the contour
							top_left = np.array(hull[hull[:, :, 1].argmin()][0])
							# cv2.circle(demo, top_left, 5, c, 2)
							# cv2.putText(demo, '  top left ', top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

							bottom_left = np.array(hull[hull[:, :, 1].argmax()][0])
							cv2.circle(demo, bottom_left, 5, c, 2)
							# cv2.putText(demo, '  bottome left ', bottom_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

							# # left_most = np.array(hull[hull[:, :, 0].argmin()][0])
							# # cv2.putText(demo, '  left most ', left_most, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

							# draw lines for demo
							# cv2.line(demo, left_hinge, top_left, g, 2)
							left_vert = (cx_mask-33,cy_mask-200)
							cv2.line(demo, left_hinge, left_vert, y, 2)
							cv2.line(demo, left_hinge, bottom_left, g, 2)

							# find vectors
							left_v1 = np.asarray(bottom_left) - np.asarray(left_hinge) # vector along the upper edge
							left_v2 = np.asarray(left_vert) - np.asarray(left_hinge) # vector along the lower edge

							# find angle using dot product of two vectors
							left_angle = 0 # initialise
							left_cosine_angle = np.dot(left_v1, left_v2) / (np.linalg.norm(left_v1) * np.linalg.norm(left_v2))
							left_angle = np.arccos(left_cosine_angle)
							left_angle = np.degrees(left_angle) # converts from radians to degrees
							# print("left angle = " + str(left_angle))

						# for right wing
						elif cx >= cx_mask:
							top_right = np.array(hull[hull[:, :, 1].argmin()][0])
							# cv2.circle(demo, top_right, 5, c, 2)
							# cv2.putText(demo, '  top right ', top_right, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

							bottom_right = np.array(hull[hull[:, :, 1].argmax()][0])
							cv2.circle(demo, bottom_right, 5, c, 2)
							# cv2.putText(demo, '  bottom right ', bottom_right, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)						

							# right_most = np.array(hull[hull[:, :, 0].argmax()][0])
							# cv2.putText(demo, '  right most ', right_most, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

							# draw lines for demo
							# cv2.line(demo, right_hinge, top_right, g, 2)
							right_vert = (cx_mask+33,cy_mask-200)
							cv2.line(demo, right_hinge, right_vert, y, 2)
							cv2.line(demo, right_hinge, bottom_right, g, 2)

							# find vectors
							right_v1 = np.asarray(bottom_right) - np.asarray(right_hinge)
							right_v2 = np.asarray(right_vert) - np.asarray(right_hinge)

							# find angle using dot product of two vectors
							right_angle = 0 # initialise
							right_cosine_angle = np.dot(right_v1, right_v2) / (np.linalg.norm(right_v1) * np.linalg.norm(right_v2))
							right_angle = np.arccos(right_cosine_angle)
							right_angle = np.degrees(right_angle)
							# print("right angle = " + str(right_angle))

			# delta wingbeat amplitude
			if right_angle == 0 or left_angle == 0:
				delta_angle = 0
			else:
				delta_angle = (left_angle) - (right_angle)

			## HAVE TO APPLY SMOOTHING FILTER TO DELTA ANGLE OUPUT (NOT DONE)

			# output values to rasp pi (-1,0,1)
			if delta_angle > 20:
				pole = 1
			elif delta_angle < -20:
				pole = -1
			else:
				pole = 0
			
			print(str(delta_angle))
			# print(str(pole))
			# print(str(left_angle)+' - '+str(right_angle)+' = '+str(delta_angle)+ ' ('+ str(pole)+ ')' +'\n')
			# print(" ")
			fn.write(str(left_angle)+','+str(right_angle)+','+str(delta_angle)+','+str(pole)+'\n')

			# reset angle to 0
			right_angle = 0
			left_angle = 0


			# demo=frame.copy()
			# if len(contours)>=1:
			# 	# print('number of contours: ',len(contours))
				
			# 	# if log_flag==1:
			# 	# 	log.write(str(cap.get(cv2.CAP_PROP_POS_FRAMES))+',')

			# 	for i in range(len(contours)):

			# 		# print( cx,cy )
			# 		# if log_flag==1:
			# 		# 	log.write(str(cx)+','+str(cy)+',')
					
			# 		# cv2.drawContours(demo,[i],-1,(0,255,0),2)

			# 		cnt = contours[i]
			# 		M = cv2.moments(cnt)					
			# 		area = cv2.contourArea(cnt)

			# 		if area>2000:


			# 			cx = int(M['m10']/M['m00'])
			# 			cy = int(M['m01']/M['m00'])		
										
			# 			# print(i,area)
			# 			cv2.circle(demo,(cx,cy),7,(0,0,255),-1)
						
			# 			ellipse = cv2.fitEllipse(cnt)
			# 			cv2.ellipse(demo,ellipse,(255,0,0),2)

			# 			# rect = cv2.minAreaRect(cnt)
			# 			# box = cv2.boxPoints(rect)
			# 			# box = np.int0(box)
			# 			# cv2.drawContours(demo,[box],0,(0,0,255),2)

			# 			rows,cols = demo.shape[:2]
			# 			[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
			# 			# lefty = int((-x*vy/vx) + y)
			# 			# righty = int(((cols-x)*vy/vx)+y)
			# 			# cv2.line(demo,(cols-1,righty),(0,lefty),(0,255,0),2)
			# 			dx = cx+np.cos(np.arctan2(vy,vx))*100
			# 			dy = cy+np.sin(np.arctan2(vy,vx))*100
			# 			cv2.line(demo,(cx,cy),(dx,dy),(0,255,0),2)

####################################################################################################################
			cv2.imshow('frame', frame)			

			cv2.imshow('calib', gray) # for calibration
			# cv2.setMouseCallback('calib', click_event)

			# cv2.imshow('threshold', thr)
			# cv2.imshow('erosion', erosion)
			# cv2.imshow('dilation', dilation)

			# cv2.imshow('roi_gray', roi_gray)	
			cv2.imshow('roi_thr', roi_thr)
			
			cv2.imshow('demo', demo) 
			# cv2.setMouseCallback('demo', click_event) # used for calibration, click on 'demo' window

			# video_output.write(rect)

			# pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
			# print (str(pos_frame)+" frames", end='\r')
		# else:
		# 	# The next frame is not ready, so we try to read it again
		# 	cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
		# 	print ("frame is not ready")
		# 	# It is better to wait for a while for the next frame to be ready
		# 	cv2.waitKey(1000)

		fps = 100 # plug => int(1000/fps) into waitKey
		if cv2.waitKey(0) == 27: # 27 is the ASCII 'Esc' key, press to exit the video
			break
		# if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
		# 	# If the number of captured frames is equal to the total number of frames,
		# 	# we stop
		# 	break
	fn.close()
	cap.release()
	# video_output.release()
	cv2.destroyAllWindows()