Efficient Object Detection using YOLOv9 Architecture
Abstract

YOLOv9 is the advanced object detection technology and it is the latest iteration of the YOLO (You Only Look Once). YOLOv9 delivers top notch performance in detecting objects, setting a standard for accuracy and speed. This new version introduces innovative methods such as Programmable Gradient Information (PGI) and Generalized Efficient Layer Aggregation Network (GELAN) to effectively address issues related to information loss and computational efficiency. 
The training and inference were performed using the CARLA_v20 dataset, with optimizations in hyper-parameters and detection thresholds. We developed an algorithm for detecting and cropping only specified classes ("bike" and "traffic_sign_90"). The results demonstrated improved object detection with enhanced precision and recall, showcasing the effectiveness of YOLOv9 in practical applications.
Acknowledgment

We express our gratitude to our mentor, Dr. Chalavadi Vishnu, for his guidance and support throughout this project. Dr. Vishnu sir’s insightful feedback, prompt responses to our queries, and constant encouragement greatly enhanced our understanding and application of YOLOv9 and related technologies. We also appreciate his patience and willingness to share his extensive knowledge, which was crucial for overcoming challenges and achieving the project’s objectives
About the Institute

Established in 2015, IIT Tirupati is an autonomous institution under the
Ministry of Education, Government of India, and is recognized as an Institute
of National Importance. Initially operating from a temporary campus, IIT
Tirupati moved to its permanent campus in Yerpedu on August 1, 2022.
Spanning 548.3 acres, the campus includes academic, hostel, housing, and
recreational zones, with ongoing construction to accommodate the growing
student body.

IIT Tirupati offers B.Tech programs in Civil, Computer Science, Electrical,
Mechanical, and Chemical Engineering, along with M.Tech, M.Sc in
Mathematics, and an MPP (Master of Public Policy). Research programs in
MS and Ph.D. have been available since 2017, with a curriculum that
emphasizes both theoretical knowledge and practical applications.

The Department of Computer Science and Engineering, headed by Dr. Sridhar
Chimalakonda, is rapidly growing and offers B.Tech, M.Tech, MS (Research),
and Ph.D. programs. The M.Tech program focuses on Data Science and
Systems. The department is active in research areas such as AI, machine
learning, algorithms, computer networks, and software engineering, with
industry collaborations including Toshiba R&D, Facebook, Bosch R&D, and
Accenture Labs. The department aims to expand its expertise in Data Science
& Systems and Cybersecurity to meet national needs.

List of Figures
	Programmable Gradient Information (PGI)
	Resultant Inference Images
	Resultant Plots after Training
	Confidence Threshold vs. F1 Score Plot
	Precision-Recall Curve (PR_Curve)
	Precision Curve (P_Curve)
	Recall Curve (R_Curve)
	Class-wise Bar Chart
	Bounding Box Distribution Plot
	Central Coordinate Distribution Plot
	Bounding Box Size Distribution Plot
	Example Output Image with Bounding Boxes
	Cropped Images
	Object Detection with Cropping 


List of Abbreviations
	YOLOv9: You Only Look Once version 9
	PGI: Programmable Gradient Information
	GELAN: Generalized Efficient Layer Aggregation Network
	IoU: Intersection over Union
	mAP: Mean Average Precision
	PR Curve: Precision-Recall Curve
Introduction and Background
YOLOv9 is the latest iteration of the YOLO series, renowned for its high accuracy and speed in object detection. YOLOv9 stands out as a significant development in Object Detection, created by Chien-Yao Wang and his team. This project explores the implementation of YOLOv9, utilizing advancements like PGI and GELAN to mitigate data loss and enhance gradient reliability, thereby improving object detection performance.  
The YOLOv9 framework introduces an innovative approach to addressing the core challenges in object detection through deep learning, mainly focusing on the issues of information loss and efficiency in network architecture. This approach has four key components: The Information Bottleneck Principle, Reversible Functions, Programmable Gradient Information (PGI), and the Generalized Efficient Layer Aggregate Network (GELAN).
Information Bottleneck Principle:

The Information Bottleneck Principle highlights how information loss occurs as data transforms within a neural network. The equation of Information Bottleneck represents the loss of mutual information between the original and transformed data by going through the deep network layers.

Information Bottleneck Principle mathematically represented as:
                       
Here: 
I  =  Mutual Information.
f , g  =  Transformation functions with parameters θ and ϕ, respectively.This represents different layers of the deep learning model. This loss can lead to Unreliable Gradients.       
![image](https://github.com/user-attachments/assets/540d59cc-42e2-4d0a-8a17-c8d0e9bbdecb)
Programmable Gradient Information (PGI):
PGI acts like a bridge, it ensures that essential data is preserved throughout the network's depth.
This leads to : 
More reliable gradient generation : Gradients are crucial for guiding the learning process in neural networks. With PGI, the gradients are based on more information, leading to better model learning.
Enhanced model convergence : As the model learns more effectively with accurate gradients, it converges faster, achieving optimal performance in a shorter time.
PGI ensures that crucial information is preserved throughout the network, essential for accurate object detection.This is achieved by introducing an Auxiliary Reversible Branch alongside the main processing path.This branch allows the model to “remember” and utilize vital information that might otherwise be lost.This leads to the generation of reliable gradients. These gradients, in turn, guide the model’s learning process more effectively, resulting in improved overall detection performance.
![image](https://github.com/user-attachments/assets/c81de7e8-c849-49ef-94ca-3dcbb1179dd4)
PGI develops by integrating three components, each serving a distinct but interrelated function within the model’s architecture. 


Main Branch: 
The main branch is optimized for inference, ensuring the model remains lean and efficient during this critical phase. It’s designed to bypass the need for auxiliary components during inference and maintain high performance without additional computational overhead.
Auxiliary Reversible Branch: 
The Auxiliary branch ensures the generation of reliable gradients and facilitates precise parameter updates. By harnessing reversible architecture, it overcomes the inherent information loss in deep network layers and enables the preservation and utilization of complete data for learning. This branch’s design allows it to seamlessly integrate or remove, ensuring that model depth and complexity do not impede inference speed.
Multi-Level Auxiliary Information: 
This method uses special networks to combine gradient information throughout the model’s layers. It tackles the problem of losing information in deep supervision models, ensuring the model fully understands the data. This technique helps it make better predictions for objects of different sizes.
Problem Statement
In the realm of deep learning, particularly in the context of object detection, one of the critical challenges is the effective transmission and retention of information through the layers of a neural network. As data passes through the network's layers, it often undergoes transformations that can lead to information loss, a phenomenon known as the Information Bottleneck. This information loss can result in unreliable gradients, which are essential for optimizing the learning process in neural networks.
Unreliable gradients can cause the model to converge slowly or even get stuck in suboptimal solutions, leading to poor detection performance, particularly when identifying complex or small objects within images. As object detection models like YOLO (You Only Look Once) become deeper and more complex, the risk of information bottlenecks becomes increasingly significant.
YOLOv9, the latest iteration of the YOLO framework, introduces innovative mechanisms to address these challenges. One of the most notable advancements in YOLOv9 is the introduction of Programmable Gradient Information (PGI). PGI is designed to preserve essential data throughout the network's depth, thereby ensuring that gradients remain reliable and the model can learn more effectively. By maintaining the integrity of gradient information, PGI facilitates faster convergence of the model and improves overall detection accuracy.

Current Approach
a. Dataset Selection and Preprocessing
The project began by selecting the CARLA_v20 dataset, a comprehensive dataset containing images of various traffic-related objects, such as vehicles, bikes, motorbikes, traffic lights, and traffic signs. This dataset is well-suited for training an object detection model like YOLOv9 because it offers a diverse range of object classes, different lighting conditions, and varying perspectives, which are crucial for creating a robust detection model.
b. Preprocessing Steps:
•	Data Loading: The CARLA_v20 dataset was downloaded and processed using the Roboflow API, which facilitated easy access to the images and annotations.
c. Model Training
The core of the approach involved training the YOLOv9 model using the pre-processed CARLA_v20 dataset.
Key Steps in Model Training:
•	Google Colab Integration: The training was conducted on Google Colab, leveraging its GPU capabilities to speed up the process. The Google Drive was mounted to save the results and checkpoints during and after the training.
•	Using Pre-trained Weights: We used pre-trained weights from the official YOLOv9 repository to initialize the model. This approach, known as transfer learning, allows the model to start with a good understanding of general object features, thereby reducing the training time required to achieve high accuracy.
•	Cloning the YOLOv9 Repository: The YOLOv9 repository was cloned from GitHub, which provided the necessary scripts and utilities for training, inference, and evaluation.
•	Training Configuration: The training was configured with a batch size of 16, and due to limited GPU availability, the number of epochs was set to 5. Although reducing the number of epochs might typically limit the learning potential of the model, this compromise was necessary to manage resource constraints.
•	Result obtained from the inference on test images:
![image](https://github.com/user-attachments/assets/97dd9a0b-36d5-4d77-8ae0-159639793f8f)
![image](https://github.com/user-attachments/assets/4b7c0eab-2d8c-4a4d-bc70-d40170c8e935)
d. Evaluation and Analysis
The performance of the YOLOv9 model was evaluated using various metrics and visual tools:
•	Class-wise Accuracy Metrics: These metrics provided insight into how well the model performed across different object classes, focusing on precision, recall, and mAP (Mean Average Precision).
![image](https://github.com/user-attachments/assets/79a0f619-5ab2-4566-803b-a38b6e587517)
![image](https://github.com/user-attachments/assets/f89d605e-0e6c-4934-a830-edc22f190574)
•	Confusion Matrix: A confusion matrix was generated to analyze the accuracy of the model in predicting each class. It helped identify classes where the model was more prone to making errors, such as confusing similar objects.
In a confusion matrix, rows represent the actual classes or labels of the data, while columns represent the predicted classes by the model. Each cell of the matrix represents the count of instances that belong to a particular combination of actual and predicted classes.
![image](https://github.com/user-attachments/assets/98a64ca3-c750-4ff9-84c8-c22b0bf582c2)
•	Performance Curves:
o	F1 Confidence Curve: This curve plotted the F1 score against different confidence thresholds, helping to identify the optimal threshold for balancing precision and recall.
      
![image](https://github.com/user-attachments/assets/cf1811b7-4af9-42f0-9269-5b60ee526484)
![image](https://github.com/user-attachments/assets/38561b0d-a10d-4535-b889-c3596df92a0e)
	Right upper plot shows the spatial distribution of bounding boxes of the different classes.
	Left bottom plot shows the distribution of the central coordinate of bounding boxes in the dataset. X-Axis represents the x-coordinate of the bounding box. Y-Axis represents the y-coordinate of the bounding box. This helps to understand where objects typically locate or appear in the image.
	Right bottom plot shows the distribution of bounding box widths and heights. This helps to understand the size variations of the objects.
e. Hyperparameter Tuning
After the initial model training, we focused on hyperparameter tuning to optimize the model's performance.
Key Hyperparameters Tuned:
•	Confidence Threshold: Initially set at 0.5, this threshold determines the minimum confidence level required for the model to classify an object. We later adjusted it to 0.7, aiming to increase the model's sensitivity and ensure it does not miss potential detections.
•	IoU (Intersection Over Union) Threshold: This metric measures the overlap between predicted bounding boxes and ground truth boxes. The IoU threshold was carefully adjusted to balance precision and recall, ensuring that the model accurately localizes objects without generating too many false positives.
f. Object Detection and Cropping
With the model trained, the next step involved using it for object detection and cropping, particularly focusing on specific classes such as 'bike' and 'traffic_sign_90'.
Object Detection Workflow:
•	Model Inference: The YOLOv9 model was applied to a set of test images to evaluate its performance in detecting objects. The inference was performed using custom-trained weights tailored to the CARLA_v20 dataset.
•	Class-specific Object Cropping: For a more detailed analysis, we implemented a process to detect, crop, and save images of specific objects. Only objects belonging to the classes 'bike' and 'traffic_sign_90' were cropped from the test images.
o	Directory Setup: A new directory named cropped_images_final_21 was created to store these cropped images.

o	Image Annotation: Each cropped image was annotated with bounding boxes and class labels, providing visual confirmation of the model's detection capability.
o	Saving and Displaying Results: The cropped images were saved with filenames indicating their class and instance count (e.g., bike_1.png, traffic_sign_90_2.png). The original images were also displayed with annotations showing bounding boxes and confidence scores.

![image](https://github.com/user-attachments/assets/0314a6e7-1796-4fac-b110-7d8879507e4e)
g. Computational Efficiency
The computational efficiency of the model was a key consideration throughout the project. The speed of inference was broken down into three main components:
•	Preprocessing Time: The time taken to prepare the image for input into the YOLO model.
•	Inference Time: The time required by the model to process the input image and produce predictions.
•	Post-Processing Time: The time taken to convert the model's output into a human-readable format (e.g., drawing bounding boxes and labels on the image).
Contributions
This project involved significant contributions to both the technical and practical aspects of object detection using the YOLOv9 framework. Below are the key contributions:
•	The project involved the successful implementation and training of the YOLOv9 model on the CARLA_v20 dataset, a task that required a deep understanding of both the dataset and the model architecture.
	Hyperparameter tuning was a critical part of the project, where the team experimented with different settings to optimize model performance. This included:
•	Adjusting Confidence and IoU Thresholds: The confidence threshold and IoU threshold were fine-tuned to strike a balance between precision and recall, enhancing the model's accuracy in detecting objects.
•	Reducing Epochs for Resource Efficiency: The team made strategic decisions to reduce the number of epochs from 25 to 5, balancing between computational limitations and model performance. This was particularly important given the constraints of running on limited GPU resources.
	One of the significant contributions was the implementation of an object-specific cropping technique. The team:
•	Developed a Custom Object Cropping Pipeline: This pipeline focused on detecting specific object classes (like 'bike' and 'traffic_sign_90'), cropping them from images, and saving the results. This approach is particularly valuable for applications where certain objects are of greater interest and need to be isolated for further analysis or processing.
	The team contributed by conducting a detailed analysis of the model's performance across different object classes:
•	Class-wise Accuracy Metrics: By evaluating metrics such as precision, recall, and mAP for each class, the team identified strengths and weaknesses in the model's ability to detect various objects. This analysis was crucial for understanding where the model excelled and where it needed improvement.
	The work done in this project lays the groundwork for future research and development:
•	Exploring Advanced Object Detection Techniques: The findings and methodologies developed can serve as a basis for exploring more advanced object detection techniques, potentially leading to even more accurate and efficient models.
•	Real-World Applications: The techniques and models developed during this project can be applied to real-world scenarios, such as autonomous driving, traffic monitoring, and other areas where robust object detection is critical.
Learnings
The YOLOv9 project was a rich learning experience, offering insights into both the theoretical and practical aspects of modern object detection technologies. Below are the key learnings gained throughout the course of the project:
	One of the most significant learnings was a deep dive into the YOLOv9 architecture, which represents the latest advancement in the YOLO series. Gained a comprehensive understanding of:
Programmable Gradient Information (PGI): This novel approach was crucial in maintaining critical data throughout the network’s depth, ensuring that essential information was preserved during the training process. Understanding PGI helped the team appreciate how this innovation addresses issues related to gradient reliability and model convergence.
	The project provided an opportunity to explore the Information Bottleneck Principle in depth:
Impact on Neural Networks: Learned how this principle explains the loss of mutual information as data is transmitted through deep network layers, leading to potential issues like unreliable gradients. This understanding was critical for grasping why innovations like PGI are necessary in advanced neural network architectures.
	The practical experience of training the YOLOv9 model on the CARLA_v20 dataset was invaluable:
Dataset Preparation and Handling: Learned how to effectively prepare and manage datasets for training, including the importance of proper dataset annotation and splitting (training, validation, and test sets).
	Hyperparameter tuning was a crucial aspect of the project that provided deep insights into:
Impact of Hyperparameters on Model Performance: Learned how different hyperparameters, such as batch size, learning rate, confidence threshold, and IoU threshold, directly influence the performance and accuracy of the model.
	The project emphasized the importance of comprehensive model evaluation:
Class-wise Accuracy Metrics: Learned how to analyze and interpret class-wise accuracy metrics, such as precision, recall, mAP50, and mAP50-95, which are essential for understanding the model’s strengths and weaknesses across different object classes.
Performance Curves and Confusion Matrix: The generation and analysis of performance curves (F1, PR, precision-confidence, recall-confidence) and confusion matrices provided deeper insights into how well the model distinguishes between different classes, which is critical for fine-tuning and improving the model.
	A key practical learning was the implementation of object-specific detection and cropping:
Developing a Targeted Detection Pipeline: Learned how to create a pipeline that focuses on detecting specific object classes, such as 'bike' and 'traffic_sign_90', and how to efficiently crop and save these objects from images. This skill is particularly useful for applications requiring focused analysis of specific objects within larger images.






Summary
This project focused on implementing and exploring YOLOv9, the latest advancement in object detection technology, which builds on the YOLO (You Only Look Once) series. The primary goal was to train and evaluate the YOLOv9 model on the CARLA_v20 dataset to detect various traffic-related objects. The process involved training the model in Google Colab, making use of GPU resources to ensure efficient processing. Key performance metrics such as precision, recall, and mAP were used to assess the model's accuracy and effectiveness. Various performance curves were generated to visualize these results, helping to understand the model's strengths and areas for improvement.
In addition to standard object detection, the project also included a specific task of detecting and cropping objects like 'bike' and 'traffic_sign_90' from the dataset. This practical application demonstrated the model's ability to focus on particular classes within a diverse dataset. The project provided valuable insights into the workings of advanced object detection technologies and offered hands-on experience with model training, evaluation, and optimization in a real-world context.











