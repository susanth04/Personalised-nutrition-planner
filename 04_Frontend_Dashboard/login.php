<?php
require_once 'config/db.php';
require_once 'config/session.php';

// If already logged in, redirect to dashboard
if (isLoggedIn()) {
    header("Location: redirect.php");
    exit;
}

$login_error = '';

// Process form submission
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $email = trim($_POST['email']);
    $password = $_POST['password'];
    $role = $_POST['role'];
    
    // Basic validation
    if (empty($email) || empty($password)) {
        $login_error = "Please enter both email and password";
    } else {
        // Prepare a select statement
        $sql = "SELECT id, email, password, role, first_name, last_name FROM users WHERE email = :email AND role = :role";
        
        if ($stmt = $pdo->prepare($sql)) {
            // Bind variables to the prepared statement as parameters
            $stmt->bindParam(":email", $email, PDO::PARAM_STR);
            $stmt->bindParam(":role", $role, PDO::PARAM_STR);
            
            // Attempt to execute the prepared statement
            if ($stmt->execute()) {
                if ($stmt->rowCount() == 1) {
                    if ($row = $stmt->fetch()) {
                        $id = $row["id"];
                        $hashed_password = $row["password"];
                        
                        if (password_verify($password, $hashed_password)) {
                            // Password is correct, start a new session
                            session_start();
                            
                            // Store data in session variables
                            $_SESSION["user_id"] = $id;
                            $_SESSION["user_email"] = $email;
                            $_SESSION["user_role"] = $row["role"];
                            $_SESSION["user_first_name"] = $row["first_name"];
                            $_SESSION["user_last_name"] = $row["last_name"];
                            
                            // Redirect to dashboard
                            header("Location: redirect.php");
                            exit;
                        } else {
                            // Password is not valid
                            $login_error = "Invalid password";
                        }
                    }
                } else {
                    // No account found with that email
                    $login_error = "No account found with that email for the selected role";
                }
            } else {
                $login_error = "Something went wrong. Please try again later.";
            }
            
            // Close statement
            unset($stmt);
        }
    }
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Nutrition Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f5f8fa;
            font-family: Arial, sans-serif;
            padding-top: 50px;
        }
        .login-container {
            max-width: 450px;
            margin: 0 auto;
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
            padding: 30px;
        }
        .login-logo {
            text-align: center;
            margin-bottom: 30px;
        }
        .login-logo h1 {
            color: #4CAF50;
            font-weight: bold;
        }
        .nav-tabs .nav-link {
            color: #6c757d;
            font-weight: 500;
        }
        .nav-tabs .nav-link.active {
            color: #4CAF50;
            font-weight: 600;
        }
        .btn-primary {
            background-color: #4CAF50;
            border-color: #4CAF50;
        }
        .btn-primary:hover {
            background-color: #3e8e41;
            border-color: #3e8e41;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="login-container">
            <div class="login-logo">
                <h1>Nutrition Dashboard</h1>
                <p class="text-muted">Please login to access your dashboard</p>
            </div>
            
            <?php if (!empty($login_error)): ?>
                <div class="alert alert-danger"><?php echo $login_error; ?></div>
            <?php endif; ?>
            
            <ul class="nav nav-tabs mb-4" id="loginTabs" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="user-tab" data-bs-toggle="tab" data-bs-target="#user-login" type="button" role="tab" aria-controls="user-login" aria-selected="true">User Login</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="doctor-tab" data-bs-toggle="tab" data-bs-target="#doctor-login" type="button" role="tab" aria-controls="doctor-login" aria-selected="false">Doctor Login</button>
                </li>
            </ul>
            
            <div class="tab-content" id="loginTabsContent">
                <!-- User Login Form -->
                <div class="tab-pane fade show active" id="user-login" role="tabpanel" aria-labelledby="user-tab">
                    <form action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]); ?>" method="post">
                        <input type="hidden" name="role" value="user">
                        <div class="mb-3">
                            <label for="userEmail" class="form-label">Email address</label>
                            <input type="email" class="form-control" id="userEmail" name="email" required>
                        </div>
                        <div class="mb-3">
                            <label for="userPassword" class="form-label">Password</label>
                            <input type="password" class="form-control" id="userPassword" name="password" required>
                        </div>
                        <div class="mb-3 form-check">
                            <input type="checkbox" class="form-check-input" id="userRemember">
                            <label class="form-check-label" for="userRemember">Remember me</label>
                        </div>
                        <button type="submit" class="btn btn-primary w-100">Login</button>
                    </form>
                </div>
                
                <!-- Doctor Login Form -->
                <div class="tab-pane fade" id="doctor-login" role="tabpanel" aria-labelledby="doctor-tab">
                    <form action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]); ?>" method="post">
                        <input type="hidden" name="role" value="doctor">
                        <div class="mb-3">
                            <label for="doctorEmail" class="form-label">Email address</label>
                            <input type="email" class="form-control" id="doctorEmail" name="email" required>
                        </div>
                        <div class="mb-3">
                            <label for="doctorPassword" class="form-label">Password</label>
                            <input type="password" class="form-control" id="doctorPassword" name="password" required>
                        </div>
                        <div class="mb-3 form-check">
                            <input type="checkbox" class="form-check-input" id="doctorRemember">
                            <label class="form-check-label" for="doctorRemember">Remember me</label>
                        </div>
                        <button type="submit" class="btn btn-primary w-100">Login</button>
                    </form>
                </div>
            </div>
            
            <div class="mt-4 text-center">
                <p>Don't have an account? <a href="register.php">Register here</a></p>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html> 