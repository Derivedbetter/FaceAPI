from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image, ImageDraw, ImageFont
# You will need an API to access cognitive services.
# If you need more information about configurations or implementing the sample code, visit the AWS docs:
# https://aws.amazon.com/developers/getting-started/python/
import awswrangler.secretsmanager as sm


API_Key = sm.get_secret_json("FACEAPI").get('API_Key')
EndPoint = "https://faceapims548.cognitiveservices.azure.com/"


Face_Client = FaceClient(EndPoint, CognitiveServicesCredentials(API_Key))

picture = open('./Images/img1.jpg', 'rb')
response = Face_Client.face.detect_with_stream(image=picture,
                                               detection_model='detection_01',
                                               recognition_model='recognition_04',
                                               return_face_attributes=['age', 'emotion'],
                                               )
img = Image.open(picture)
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("arial.ttf", 15)
for face in response:
    age = face.face_attributes.age
    emotion = face.face_attributes.emotion
    happiness = '{0:.0f}%'.format(emotion.happiness * 100)
    anger = '{0:.0f}%'.format(emotion.anger * 100)
    sadness = '{0:.0f}%'.format(emotion.sadness * 100)
    # Draw rectangles for faces
    rect = face.face_rectangle
    left = rect.left
    top = rect.top
    right = rect.width + left
    bottom = rect.height + top
    draw.rectangle(((left, top), (right, bottom)), outline="red", width=2)
    # Print percentages of emotions
    draw.text((right + 4, top), "Age: " + str(int(age)), fill=(0, 0, 0), font=font)
    draw.text((right + 4, top + 40), "Happy: " + happiness, fill=(0, 128, 0), font=font)
    draw.text((right + 4, top + 80), "Sad: " + sadness, fill=(0, 0, 255), font=font)
    draw.text((right + 4, top + 120), "Angry: " + anger, fill=(255, 0, 0), font=font)


img.show()
