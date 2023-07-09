const imageUploadInput = document.getElementById('imageUpload');
const imagePreviewDiv = document.getElementById('imagePreview');
const uploadButton = document.getElementById('uploadButton');

//event listener for image upload input change
imageUploadInput.addEventListener('change' , function() {
    const imageFile = imageUploadInput.files[0];

    //display img preview
    const reader = new FileReader();
    reader.onload = function(e) {
        const imagePreview = document.createElement('img');
        imagePreview.src = e.target.result;
        imagePreviewDiv.innerHTML = '';
        imagePreviewDiv.appendChild(imagePreview);
    };
    reader.readAsDataURL(imageFile);

    //enable upload btn
    uploadButton.disabled = false;

});

//event listener for upload button
uploadButton.addEventListener('click', function() {
    //get selected img file
    const imageFile = imageUploadInput.files[0];

    //perform upload
    uploadImage(imageFile);
});

//function to upload image to server
function uploadImage(imageFile) {
    const fileInput = document.getElementById('imageUpload')
    const file =  fileInput.files[0];
    console.log('Selected file:' , file);
    const reader = new FileReader();
  reader.onload = function(event) {
    const imageUrl = event.target.result;

    // Display the image
    const imageElement = document.createElement('img');
    imageElement.src = imageUrl;
    document.body.appendChild(imageElement);
  };




  reader.readAsDataURL(file);
    // perform upload logic
    //handle response from server
    //can use AJAX fetch API or library to send image to backend
}