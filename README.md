# Breast-Cancer-Segmentation
Breast Cancer Segmentation using PyTorch

The model is trained using Facebook's Segment Anything Model. The code along with the model is available in PyTorch.

The model is trained on ultrasound images of normal, benign, and malignant breast cancer. **The Model's Test IOU Accuracy is 91%.**

![Alt text](https://www.kaggleusercontent.com/kf/166274566/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..LmYtyVt-fDrXhodllfPW8g.ie7lqFyvcuSh8YK_-NztQRDSzw673FmRRuSp6ZbhVXi_eVYXbFvE06nCfcVXttm4uZBJRdGL_OwnldSfDwxlASHp4Bfbk2DJsRhfdX_IkJRCdW6LC2Ds3d29p79Hx_JpKzeQukpeJIMGygCtbdIXZ4l0wzF44qJG0yCKqGO-5YRPF21mJaMQ7ktJmqJck0Rya_PHWXPdZcqMDOB74e0KJKgTmvyNYivf-OUQcYO3osSVYhyDvPErcBfScVqFG2UviBGHA9tcOTWuk7VcszFXuYRG9UB82k8Z2nxiR0GDRkrhQON_JSxVdNnBE4n2bklw83oUCOQ2xpVfcYCU9ao2xMHJSNZAm4Q2V7Pyetv4J1s9PbRtahYJIcyfikGWffJhLfF2sHO3YchAZGu60TcEy7g08cZNjU32GdEVrZNS-G2bNvRxFKYHML0njTu3TGGmx-FAmgX3bRGf_EgDkXLqicOIOrQjweOCp6EFLJaUM15xwLaxYeZAc2X8NexOPlUGumD2kSwhUIOsghUs7JUNk2H9mTKNRw-A-QFa1TnVlzamxz3PmXPZ7eXT-GJgSJfOaJIdbIEMPjjwJrES4DPbu8B6_YQTOjQ_0N6KcRx5azRHA4rkhRTP2WWKmE9UjoCUFVXNUfMxKS1LawmI4lcW-W64ME-jNlMa1UCvv0cxFuvFSIQf-8n2IheF0dI5QM_K._0Okx9kV0oYC65W3TIzlsA/__results___files/__results___11_0.png "Optional Title")


### How to load the model:
```
# *You will need to download the model first*
model = torch.load(MODEL'S PATH)
model.eval()
```

### How to use the model for inference:
```
# *You will need to download the model first*
model = torch.load(MODEL'S PATH)
model.eval()

# get an image URL or a path
img_url = 'https://s3.amazonaws.com/static.wd7.us/9/90/Mamma_ca_1.jpg'
raw_image = Image.open(requests.get(img_url, stream = True).raw).convert('RGB')

# Apply appropriate transformations to the model
image = v2.Resize(size = (1024, 1024), antialias=True)(raw_image)
image = ToTensor()(image).unsqueeze(0)
image = v2.Normalize(mean, std)(image)
image = np.array(image).astype(np.uint8)

# Process the image and pass it to the model
inputs = processor(image, return_tensors="pt") # processor is wanglab/medsam-vit-base
outputs = model(**inputs)

# Lastly, show the image
plt.imshow(image.squeeze(0).transpose(1, 2, 0))
plt.axis('off')
plt.show()
```

You should see something like this:

![Alt text](https://www.kaggleusercontent.com/kf/166274566/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..RYbxEt03tpGxBDLAyLo7Dw.dN4oVi-0gVFvC5iarMqxVEADBAGCQu5XqaT-twM6clEM9o3YrhqPepFpopWXhmiiFadWm9xIQyN-mxbf5Amu945CN60rt06866jv9EZR1Yn2k3Ihm3s-JUS3XhUdf8oY5HJb9wUTyxAPLzVFaZqDT1R9EH3tfKxGMUNnB9WpaXhH5-CtHkjZ5XvnZJRlvcmKECBSVl4-XkPM4YdFfpM8ea1C04bzMF8qx1RWGytA6r-AOh-KGJimxE3JGGjF4JhNmEi2ROpeARtnr4rLKyCmi73ewb4_Qh0m3HoGop4oDq4EfvINKePzR9vYnqm80CidmkOnF9leotdh3GMwYHzMFNBKR_lBRrMrI_DoY67rGAVlg49SrS9CO4RONWS66hAbt8cZpvU7tJGXyjfvwLL8Aj5RczbStwaZLHYGLKH7TTVam1_wC7JS_uREWaTHBR7DUs7RL0dQnLpgylYYW_2lKPQbCZIycmbHZ_X4v4VicwWLgRC23fYIJWsVlvGnpakxlVRZ0S7Eanur0Q1XIKycpSHP-r83IajkwG39jOR4ijByB3lEZ-_hUAQkQhsFE5FoyK5ho1n1ckqoopngztTdsfgPDX4bWbV9ai7R-xY1Cxd9GAGLBfZBJ35PeOkQTm-pMC4t7oipk1lwQ4ieyudClQIBk5Vaga_1f0yBX_MeAi2DDxB8QlKREJX3uBhyXwJn.BO1V8WVud2LsPsCZhZ2PNQ/__results___files/__results___10_0.png "Optional Title")
