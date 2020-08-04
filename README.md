# VIRTUAL GLASSES TRY ON SYSTEM BY USING AUGMENTED REALITY
 
“Virtual Glasses Try on System by Using Augmented Reality” project is a glasses trial project
in virtual environment using image processing and augmented reality technologies.
Within the scope of this project, the user should be provided a photo to the system
with real time video or photo upload options. The system presents the result image to
the user by placing the selected glasses appropriately in this photo.

In this project, which can work by real time video or photo upload, face detection
will be made using machine learning functions. With the help of image processing
functions, landmarks will be found in the detected facial region. With the help of the
found landmarks, the following calculations will be made;
1
• Detecting landmarks in the face: The "dlib.shape-predictor
("shape-predictor-68-face-landmarks.dat")" function of the "dlib" library is
used for this process. The points indicated are shown in the "System Design"
section (Software Design Subsection).
• Determination of the angle of inclination of the head from right to left
(two-dimensional slope angle),
• Determination of the angle of inclination of the head from front to side
(three-dimensional slope angle),
• Determination of the border areas of the eyes,
• Determination of the eye center.
Within the scope of the project, the glasses are divided into three parts so that the
glasses fit better on the face;
1. Glass part
2. Left eyeglass handle
3. Right eyeglass handle
Using the data from the calculations in the face area, the following procedures will be
applied to the glasses photo at our project;
• Rotating according to two-dimensional degree
• Perspective rotating according to three-dimensional degree
• Resizing the picture
• Adjusting the opacity of the eyeglass using the value (“alpha”) in
non-background photos (“.png”)
• Placing the glasses photo on the face

Example outputs:

![save1](https://user-images.githubusercontent.com/34898893/89314111-efc4d180-d681-11ea-963d-6c9656e8ce3e.PNG)![beckham](https://user-images.githubusercontent.com/34898893/89314143-f6ebdf80-d681-11ea-9356-6a922c04ca71.png)
