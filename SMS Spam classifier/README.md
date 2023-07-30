# SMS SPAM Detector 
### Overview
This SMS Spam Classifier ML project aims to build a machine learning model that can accurately classify SMS messages as either "spam" or "not spam" (ham). Spam messages are unsolicited, often promoting scams or irrelevant content, while ham messages are legitimate and desired communications. The model will be trained on a labeled dataset of SMS messages to learn patterns and characteristics of spam, enabling it to make predictions on new, unseen messages.

### Dataset
The dataset used for training and evaluating the SMS spam classifier is collected from Kaggle and contains a balanced set of labeled SMS messages. It consists of two main columns:
Text: The actual SMS content.
Label: The class label indicating whether the message is "spam" or "ham."

### Model Architecture
The SMS spam classifier employs a supervised learning approach, using natural language processing (NLP) techniques to process and vectorize the SMS messages. The following steps are involved:

Text Preprocessing: The raw SMS messages are cleaned by removing punctuation, converting to lowercase, and handling stop words. Tokenization is performed to break the text into individual words or tokens.

Feature Extraction: The processed text is converted into numerical features using techniques like TF-IDF (Term Frequency-Inverse Document Frequency) or Bag of words.

Model Selection: Several machine learning models, such as Naive Bayes, Logistic Regression, or Support Vector Machines, are evaluated for their performance on the dataset.

Model Training: The chosen model is trained on the labeled dataset using a portion for training and the rest for validation.

Model Evaluation: The trained model is evaluated on a separate test set to measure its accuracy, precision, recall, F1-score, and other relevant metrics.

Model Deployment: After successful evaluation, the final model is saved and can be used for classifying new, unseen SMS messages.

Usage
Once the model is trained and ready for use, the SMS Spam Classifier can be utilized in various ways:

![Screenshot 2023-07-24 171215](https://github.com/prasadkanthuri/Portfolio/assets/135444495/f34583be-cb75-4f3e-8263-fd2007c43711)

