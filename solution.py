def face_rec(image):
    """function that recognizes the person in the given photo
    input:
      image grayscale or color image with face on it
    output;
      person_id: id of a person (class)
    """
    from matplotlib import pyplot as plt
    import numpy as np
    import cv2
    from collections import Counter
    import pickle
    from numpy import linalg as LA

    plt.rcParams["figure.figsize"] = (5, 5)  # (w, h)

    using_VJ = 0  ######## using Viola Jones for detection
    plot_and_print = 1  ############# plotting the images
    k_dist = 1  ########### number of distances taken into account in the voting for the id

    with open('trainingVariables', 'rb') as f:
        dim, dim1, mean_faces, u, W, W_total, index, ids, A = pickle.load(f)

    imgbgr = cv2.imread(image, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ########################## detecting face ####################################
    if using_VJ:
        ################ using viola Jones
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if (len(faces) == 0):
            print('error!! no face detected')
            return -1
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            if len(faces) > 1:
                return -1
    else:
        ################ using deep learning
        modelFile = "opencv_face_detector_uint8.pb"
        configFile = "opencv_face_detector.pbtxt"
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
        conf_threshold = 0.9

        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()

        detect = 0
        if detections.shape[2] == 0:
            return -1
        for k in range(detections.shape[2]):
            confidence = detections[0, 0, k, 2]
            if confidence > conf_threshold:
                box = detections[0, 0, k, 3:7] * np.array([w, h, w, h])
                roi_gray = gray[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
                roi_color = img[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
                detect = 1
        if detect == 0:
            return -1

    ######################### resizing the image face#############################
    roi = cv2.resize(roi_gray, dim1, interpolation=cv2.INTER_AREA)

    ################### projection  onto the face space ##########################
    gamma_test = roi.flatten()[:]
    phi_test = (gamma_test.transpose() - mean_faces).transpose()
    w_test = np.dot(u.transpose(), phi_test)  ## weights of each image in columns
    dist = []

    for i in range(W_total.shape[1]):
        dist.append(LA.norm(w_test - W_total[:, i]))  ### euclidean distance

    ###################### voting for the best minimum distance ################
    distance = dist
    indices = []
    for i in range(k_dist):
        ind = np.argmin(distance)
        distance.pop(ind)
        indices.append(ind)

    occur = Counter(ids[indices])
    min_occur = 0
    if len(set(ids[indices])) == 1:
        ind = indices[0]
        person_id = ids[ind]
    else:
        for id, oc in occur.items():
            if oc > min_occur:
                min_occur = oc
                person_id = id
    ind = index[person_id][0]

    if plot_and_print:
        ######################### plot the input image ##############################""
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 4, 1)
        plt.imshow(img, 'gray')
        plt.title('input image'.format())
        plt.xticks([]), plt.yticks([])

        ########################  plotting face ######################################
        plt.subplot(1, 4, 2)
        plt.imshow(roi_color, 'gray')
        plt.title('detected face'.format())
        plt.xticks([]), plt.yticks([])

        ################## plotting identified and reconstructed face ################
        plt.subplot(1, 4, 3)
        plt.imshow((A[:, ind] + mean_faces).reshape(dim), 'gray')
        plt.title('identified face')
        plt.xticks([]), plt.yticks([])

        A_rec = np.sum(np.multiply(u, w_test), axis=1) + mean_faces
        plt.subplot(1, 4, 4)
        plt.imshow(A_rec.reshape(dim), 'gray')
        plt.title('reconstruted face ')
        plt.xticks([]), plt.yticks([])
        plt.show()

        #################### printing distance within and from face space ############
        difs = dist[ind]
        dffs = LA.norm(np.sum(np.multiply(u, w_test), axis=1) - phi_test)
        print('the distance within the face space is difs = :{}'.format(difs))
        print('the distance from the face space is dffs = :{}'.format(dffs))
        print('the face identified is id = {}'.format(person_id))

    return person_id


