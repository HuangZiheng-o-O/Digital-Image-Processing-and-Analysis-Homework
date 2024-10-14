// let originalImage = null;
//
// function uploadImage() {
//     const imageInput = document.getElementById('imageInput');
//     if (imageInput.files.length === 0) {
//         alert("请上传图像。");
//         return;
//     }
//
//     const formData = new FormData();
//     formData.append('image', imageInput.files[0]);
//
//     fetch('/process', {
//         method: 'POST',
//         body: formData
//     })
//     .then(response => response.json())
//     .then(data => {
//         originalImage = new Image();
//         originalImage.src = 'data:image/jpeg;base64,' + btoa(data.result); // 修复为 base64
//
//         // 设置点击事件以获取种子点
//         originalImage.onclick = function(event) {
//             const x = event.offsetX;
//             const y = event.offsetY;
//             processImageWithSeed(x, y);
//         };
//
//         const imageContainer = document.getElementById('imageContainer');
//         imageContainer.innerHTML = ''; // 清空之前的内容
//         imageContainer.appendChild(originalImage);
//     })
//     .catch(error => console.error('Error:', error));
// }
//
// function processImageWithSeed(x, y) {
//     const seedPoint = { x: x, y: y };
//
//     const formData = new FormData();
//     formData.append('image', originalImage.src); // 注意：这里需要使用二进制数据
//     formData.append('seed_point', JSON.stringify(seedPoint));
//
//     fetch('/process', {
//         method: 'POST',
//         body: formData
//     })
//     .then(response => response.json())
//     .then(data => {
//         const outputDiv = document.getElementById('output');
//         const resultImage = new Image();
//         resultImage.src = 'data:image/jpeg;base64,' + btoa(data.result); // 返回处理后的图像
//         outputDiv.innerHTML = ''; // 清空之前的内容
//         outputDiv.appendChild(resultImage);
//     })
//     .catch(error => console.error('Error:', error));
// }
