import cv2
import numpy as np

# Loading images from the current directory (replace with yours!!!)
image = cv2.imread('/home/vedant/Desktop/uas_recruitnment/sample image 1.png')
image_blurred= cv2.GaussianBlur(image, (9,9),0) #Applying gaussian blur (trial and error)

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




#Block of code to find the total number of red houses in the image
#---------------------------------------------------------------------------------
contours_red, hierarchy= cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image_blurred,contours_red, -1 ,(0,255,0),3)
total_red= len(contours_red)
# print("total_red:",total_red)
#---------------------------------------------------------------------------------

#Block of code to find the total number of blue houses in the image
#---------------------------------------------------------------------------------
contours_blue, hierarchy= cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image_blurred,contours_blue, -1 ,(0,255,0),3)
total_blue=len(contours_blue)
# print("total_blue: ", total_blue)
#---------------------------------------------------------------------------------



#Block of code to find the number of red houses on burnt side
#-------------------------------------------------------------------------------
mask_burnt_final=np.zeros_like(image_blurred)
burnt_contours, hierarchy = cv2.findContours(burnt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(mask_burnt_final, burnt_contours, -1, (255, 255, 255), thickness=cv2.FILLED)    #creating the correct burnt masked region (without the cutout triangles)

mask_burnt_final_gray = cv2.cvtColor(mask_burnt_final, cv2.COLOR_BGR2GRAY)   

burnt_red_or=cv2.bitwise_or(red_mask,mask_burnt_final_gray)
burnt_red_or_contours, hierarchy= cv2.findContours(burnt_red_or, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print("red houses on burnt side: ", total_red-len(burnt_red_or_contours)+1)
red_burnt=total_red-len(burnt_red_or_contours)+1

#-----------------------------------------------------------------------------------


#Block of code to find the number of red houses on green side
#--------------------------------------------------------------------------------------

mask_green_final=np.zeros_like(image_blurred)
green_contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(mask_green_final, green_contours, -1, (255, 255, 255), thickness=cv2.FILLED)

mask_green_final_gray = cv2.cvtColor(mask_green_final, cv2.COLOR_BGR2GRAY)  #creating the correct burnt masked region (without the cutout triangles)

green_red_or=cv2.bitwise_or(red_mask,mask_green_final_gray)
green_red_or_contours, hierarchy= cv2.findContours(green_red_or, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print("red houses on green side: ",total_red-len(green_red_or_contours)+1)
red_green= total_red-len(green_red_or_contours)+1
#--------------------------------------------------------------------------------------------



#Block of code to find the number of blue houses on burnt side
#-------------------------------------------------------------------------------

mask_burnt_final=np.zeros_like(image_blurred)
burnt_contours, hierarchy = cv2.findContours(burnt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(mask_burnt_final, burnt_contours, -1, (255, 255, 255), thickness=cv2.FILLED)

mask_burnt_final_gray = cv2.cvtColor(mask_burnt_final, cv2.COLOR_BGR2GRAY)  #creating the correct burnt masked region (without the cutout triangles)

burnt_blue_or = cv2.bitwise_or(blue_mask, mask_burnt_final_gray)  

burnt_blue_or_contours, hierarchy= cv2.findContours(burnt_blue_or, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image_blurred,burnt_blue_or_contours, -1, (255,0,0),3)
# print("blue houses on burnt side: ", total_blue-len(burnt_blue_or_contours)+1+total_red-len(green_red_or_contours)+1)    #if it works it works
blue_burnt=total_blue-len(burnt_blue_or_contours)+1+total_red-len(green_red_or_contours)+1
#-----------------------------------------------------------------------------------



#Block of code to find the number of blue houses on green side
#--------------------------------------------------------------------------------------

mask_green_final=np.zeros_like(image_blurred)
green_contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(mask_green_final, green_contours, -1, (255, 255, 255), thickness=cv2.FILLED)

mask_green_final_gray = cv2.cvtColor(mask_green_final, cv2.COLOR_BGR2GRAY)  #creating the correct burnt masked region (without the cutout triangles)

green_blue_or = cv2.bitwise_or(blue_mask, mask_green_final_gray) 
green_blue_or_contours, hierarchy= cv2.findContours(green_blue_or, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print("blue houses on green side: ",total_blue-len(green_blue_or_contours)+1)
blue_green=total_blue-len(green_blue_or_contours)+1
#----------------------------------------------------------------------------------------------




# Applied colors to the binary masks
output_image = np.zeros_like(image_blurred)
output_image[burnt_mask > 0] = [175, 238, 238]   # Sand yellow for burnt grass
output_image[green_mask > 0] = [194, 178, 128]   # Pale blue for green grass
output_image[red_mask > 0] = [0, 0, 255]         # Red for red houses
output_image[blue_mask > 0] = [255, 0, 0]        # Blue for blue houses



# You can remove the commeted code to see the seperate masks for diagnosis

cv2.imshow('Masked Image', output_image)
# cv2.imshow('redmask',red_mask)
# cv2.imshow('burntmask',burnt_mask)
# cv2.imshow('totalmask',total_mask)
# cv2.imshow("pray",mask_burnt_final_gray)
# cv2.imshow("prayyyy",red_burnt_mask)
# cv2.imshow('bluemask',blue_mask)
# cv2.imshow('red_on_burnt_side',red_burnt_side)
# cv2.imshow('image',image_blurred)


# Coding the required expected output
burnt_green_houses_list=[[blue_burnt+red_burnt],[blue_green+red_green]]
Pb=(blue_burnt*2)+(red_burnt*1) #total priority of houses on the burnt grass 
Pg=(blue_green*2)+(red_green*1) #total priority of houses on the green grass
burnt_green_house_priority=[[Pb],[Pg]]
rescue_ratio=[[Pb//Pg]]

print("burnt_green_house list: ", burnt_green_houses_list) 
print("burnt_green_priority_list: ", burnt_green_house_priority)
print("rescue_ratio: ", rescue_ratio)





# Wait for a key press, and close if 'q' is pressed
while True:
    key = cv2.waitKey(1)  # Wait for 1 millisecond
    if key == ord('q'):    # If 'q' is pressed
        break

cv2.destroyAllWindows()