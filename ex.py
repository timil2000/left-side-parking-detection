import cv2

#establish capture
cap=cv2.VideoCapture('1.mov')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
result = cv2.VideoWriter('Save.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         cv2.CAP_PROP_FRAME_COUNT, size)
#print(cv2.CAP_PROP_FRAME_COUNT)
#loop through each frame
while(cap.isOpened()):
  ret,frame=cap.read()
  #cv2.imshow("video",cap)
  if ret == True:
      count = 0
      point1 = (0 , 1700)  # Ending Point
      point2 = (3840 , 1700)  # Line Color in BGR Format
      color = (255, 0 , 0)  # will be Blue
      thickness = 10
      linetype = cv2.LINE_AA

      image = cv2.line(frame, point1, point2, color, thickness, linetype)

      result.write(frame)
      # cv2.imshow("Video",frame)

      if cv2.waitKey(10) & 0xFF == ord('q'):
        break
  else:
      break

height = image.shape[0]
width = image.shape[1]
channels = image.shape[2]
print('Image Height       : ', height)
print('Image Width        : ', width)
print('Number of Channels : ', channels)
#close down everything
cap.release()
result.release()
cv2.destroyAllWindows()
