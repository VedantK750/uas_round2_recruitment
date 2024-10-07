import cv2
import numpy as np

# Loading images from the current directory (replace with yours!!!)
image = cv2.imread('/home/vedant/Desktop/uas_recruitnment/4.png')
image_blurred= cv2.GaussianBlur(image, (13,13),0) #Applying gaussian blur (trial and error)

# Converted to HSV color space for better color segmentation
hsv_image = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2HSV)

# Defined HSV range for burnt grass (brown)
lower_burnt = np.array([10, 100, 20])  
upper_burnt = np.array([20, 255, 200])  

# Defined HSV range for green grass
lower_green = np.array([40, 40, 40])  
upper_green = np.array([90, 255, 255])  

# Defined HSV range for red houses
lower_red1 = np.array([0, 120, 70])  # Lower red hue
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])  # Higher red hue
upper_red2 = np.array([180, 255, 255])

lower_blue = np.array([100, 100, 50])  
upper_blue = np.array([130, 255, 255])  


# Created masks for each component in the image
burnt_mask= cv2.inRange(hsv_image, lower_burnt, upper_burnt)
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Combined red masks 
red_mask = cv2.bitwise_or(red_mask1, red_mask2)
# Creating total mask (red + blue)
total_mask= cv2.bitwise_or(red_mask,blue_mask)




# Block of code to find the total number of red houses in the image
#---------------------------------------------------------------------------------
contours_red, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image_blurred, contours_red, -1, (0, 255, 0), 3)

# Counting for contours for specific area range
count_red_in_range = 0
for c in contours_red:
    area = cv2.contourArea(c)
    if 800 <= area <= 1400:
        count_red_in_range += 1
total_red = count_red_in_range
#---------------------------------------------------------------------------------



# Block of code to find the total number of blue houses in burnt region
#---------------------------------------------------------------------------------
contours_blue, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image_blurred, contours_blue, -1, (0, 255, 0), 3)

# Counting for contours for specefic range
count_blue_in_range = 0
for c in contours_blue:
    area = cv2.contourArea(c)
    if 800 <= area <= 1400:
        count_blue_in_range += 1
total_blue=count_blue_in_range

#----------------------------------------------------------------------------------------



#Block of code to find the number of red houses on burnt region
#-------------------------------------------------------------------------------
mask_burnt_final=np.zeros_like(image_blurred)
burnt_contours, hierarchy = cv2.findContours(burnt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(mask_burnt_final, burnt_contours, -1, (255, 255, 255), thickness=cv2.FILLED)    #creating the correct burnt masked region (without the cutout triangles)

mask_burnt_final_gray = cv2.cvtColor(mask_burnt_final, cv2.COLOR_BGR2GRAY)   #converting to grayscale ( because  mask_burnt_final still in bgr format) ---> either white or black

burnt_red_or=cv2.bitwise_or(red_mask,mask_burnt_final_gray)
burnt_red_or_contours, hierarchy= cv2.findContours(burnt_red_or, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
red_burnt=total_red-len(burnt_red_or_contours)+1

#-----------------------------------------------------------------------------------


#Block of code to find the number of red houses on green side
#--------------------------------------------------------------------------------------

red_green= total_red-red_burnt
#---------------------------------------------------------------------------------------------


#Block of code to find the number of blue houses on burnt side
#------------------------------------------------------------------------------------------------------
mask_burnt_final = np.zeros_like(image_blurred)
burnt_contours, hierarchy = cv2.findContours(burnt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(mask_burnt_final, burnt_contours, -1, (255, 255, 255), thickness=cv2.FILLED)

mask_burnt_final_gray = cv2.cvtColor(mask_burnt_final, cv2.COLOR_BGR2GRAY)

burnt_blue_and = cv2.bitwise_and(blue_mask, mask_burnt_final_gray)


burnt_blue_and_contours, hierarchy = cv2.findContours(burnt_blue_and, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image_blurred, burnt_blue_and_contours, -1, (255, 0, 0), 3)

# To find the number of contours in specefic area constraint
valid_contour_count = 0
for c in burnt_blue_and_contours:
    area = cv2.contourArea(c) 
    if 800 <= area <= 1400:
        valid_contour_count += 1 

blue_burnt=valid_contour_count

#-----------------------------------------------------------------------------------



#Block of code to find the number of blue houses on green side
#--------------------------------------------------------------------------------------
blue_green=total_blue-blue_burnt
#----------------------------------------------------------------------------------------------




# Applied colors to the binary masks
output_image = np.zeros_like(image_blurred)
output_image[burnt_mask > 0] = [175, 238, 238]   # Sand yellow for burnt grass
output_image[green_mask > 0] = [194, 178, 128]   # Pale blue for green grass
output_image[red_mask > 0] = [0, 0, 255]         # Red for red houses
output_image[blue_mask > 0] = [255, 0, 0]        # Blue for blue houses



# You can remove the commeted code to see the seperate masks for diagnosis

cv2.imshow('Masked Image', output_image)
# Coding the required expected output
burnt_green_houses_list=[[blue_burnt+red_burnt],[blue_green+red_green]]
Pb=(blue_burnt*2)+(red_burnt*1) #total priority of houses on the burnt grass 
Pg=(blue_green*2)+(red_green*1) #total priority of houses on the green grass
burnt_green_house_priority=[[Pb],[Pg]]
rescue_ratio=[[Pb/Pg]]

print("burnt_green_house list: ", burnt_green_houses_list) 
print("burnt_green_priority_list: ", burnt_green_house_priority)
print("rescue_ratio: ", rescue_ratio)





# Wait for a key press, and close if 'q' is pressed
while True:
    key = cv2.waitKey(1)  # Wait for 1 millisecond
    if key == ord('q'):    # If 'q' is pressed
        break

cv2.destroyAllWindows()