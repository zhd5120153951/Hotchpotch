// document.addEventListener('DOMContentLoaded', function () {
//     //如果只有一个submit是,可以这样
//     const loginForm = document.getElementById('loginForm');

//     loginForm.addEventListener('submit', function (event) {
//         event.preventDefault();

//         const username = document.getElementById('username').value;
//         const password = document.getElementById('password').value;
//         fetch('/login', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json'
//             },
//             body: JSON.stringify({
//                 username: username,
//                 password: password
//             })
//         })
//             .then(response => response.json())
//             .then(data => {
//                 if (data.success) {
//                     window.location.href = data.redirect;
//                 } else {
//                     alert(data.message);
//                 }
//             })
//             .catch(error => {
//                 console.error('Error:', error);
//             });
//     });
// });

// 多个提交是,分别为不同的id的submit添加
document.getElementById('login').addEventListener('click', function () {
    // 在这里处理发送消息到后端的逻辑
    // 可以使用 fetch API 或者其他 HTTP 请求方式发送请求
    // 例如使用 fetch API：
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    fetch('/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            username: username,
            password: password
        })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                window.location.href = data.redirect;
            } else {
                alert(data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
});

document.getElementById('cancel').addEventListener('click', function () {
    // 在这里处理发送消息到后端的逻辑
    // 可以使用 fetch API 或者其他 HTTP 请求方式发送请求
    // 例如使用 fetch API：
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    fetch('/cancel', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            username: username,
            password: password
        })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                window.location.href = data.redirect;
            } else {
                alert(data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
});


