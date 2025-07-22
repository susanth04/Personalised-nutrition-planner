<?php
require_once 'config/db.php';
require_once 'config/session.php';

// If already logged in, redirect to dashboard
if (isLoggedIn()) {
    header("Location: redirect.php");
    exit;
}

$register_error = '';
$success_message = '';

// Process form submission
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $email = trim($_POST['email']);
    $password = $_POST['password'];
    $confirm_password = $_POST['confirm_password'];
    $first_name = trim($_POST['first_name']);
    $last_name = trim($_POST['last_name']);
    $role = $_POST['role'];
    
    // Basic validation
    if (empty($email) || empty($password) || empty($confirm_password) || empty($first_name) || empty($last_name)) {
        $register_error = "Please fill all required fields";
    } elseif (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
        $register_error = "Please enter a valid email address";
    } elseif (strlen($password) < 6) {
        $register_error = "Password must have at least 6 characters";
    } elseif ($password != $confirm_password) {
        $register_error = "Passwords do not match";
    } else {
        // Check if email already exists
        $sql = "SELECT id FROM users WHERE email = :email";
        
        if ($stmt = $pdo->prepare($sql)) {
            $stmt->bindParam(":email", $email, PDO::PARAM_STR);
            
            if ($stmt->execute()) {
                if ($stmt->rowCount() > 0) {
                    $register_error = "This email is already registered";
                }
            } else {
                $register_error = "Something went wrong. Please try again later.";
            }
            
            unset($stmt);
        }
        
        // If no errors, proceed with registration
        if (empty($register_error)) {
            $sql = "INSERT INTO users (email, password, role, first_name, last_name) VALUES (:email, :password, :role, :first_name, :last_name)";
            
            if ($stmt = $pdo->prepare($sql)) {
                // Hash the password
                $hashed_password = password_hash($password, PASSWORD_DEFAULT);
                
                // Bind parameters
                $stmt->bindParam(":email", $email, PDO::PARAM_STR);
                $stmt->bindParam(":password", $hashed_password, PDO::PARAM_STR);
                $stmt->bindParam(":role", $role, PDO::PARAM_STR);
                $stmt->bindParam(":first_name", $first_name, PDO::PARAM_STR);
                $stmt->bindParam(":last_name", $last_name, PDO::PARAM_STR);
                
                // Execute the statement
                if ($stmt->execute()) {
                    $success_message = "Registration successful! You can now <a href='login.php'>login</a>.";
                } else {
                    $register_error = "Something went wrong. Please try again later.";
                }
                
                unset($stmt);
            }
        }
    }
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Register - Nutrition Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f5f8fa;
            font-family: Arial, sans-serif;
            padding-top: 50px;
        }
        .register-container {
            max-width: 500px;
            margin: 0 auto;
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
            padding: 30px;
        }
        .register-logo {
            text-align: center;
            margin-bottom: 30px;
        }
        .register-logo h1 {
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
        <div class="register-container">
            <div class="register-logo">
                <h1>Nutrition Dashboard</h1>
                <p class="text-muted">Create a new account</p>
            </div>
            
            <?php if (!empty($register_error)): ?>
                <div class="alert alert-danger"><?php echo $register_error; ?></div>
            <?php endif; ?>
            
            <?php if (!empty($success_message)): ?>
                <div class="alert alert-success"><?php echo $success_message; ?></div>
            <?php else: ?>
            
            <ul class="nav nav-tabs mb-4" id="registerTabs" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="user-reg-tab" data-bs-toggle="tab" data-bs-target="#user-register" type="button" role="tab" aria-controls="user-register" aria-selected="true">User Registration</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="doctor-reg-tab" data-bs-toggle="tab" data-bs-target="#doctor-register" type="button" role="tab" aria-controls="doctor-register" aria-selected="false">Doctor Registration</button>
                </li>
            </ul>
            
            <div class="tab-content" id="registerTabsContent">
                <!-- User Registration Form -->
                <div class="tab-pane fade show active" id="user-register" role="tabpanel" aria-labelledby="user-reg-tab">
                    <form action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]); ?>" method="post">
                        <input type="hidden" name="role" value="user">
                        <div class="row mb-3">
                            <div class="col">
                                <label for="userFirstName" class="form-label">First Name</label>
                                <input type="text" class="form-control" id="userFirstName" name="first_name" required>
                            </div>
                            <div class="col">
                                <label for="userLastName" class="form-label">Last Name</label>
                                <input type="text" class="form-control" id="userLastName" name="last_name" required>
                            </div>
                        </div>
                        <div class="mb-3">
                            <label for="userEmail" class="form-label">Email address</label>
                            <input type="email" class="form-control" id="userEmail" name="email" required>
                        </div>
                        <div class="mb-3">
                            <label for="userPassword" class="form-label">Password</label>
                            <input type="password" class="form-control" id="userPassword" name="password" required>
                            <div class="form-text">Password must be at least 6 characters long.</div>
                        </div>
                        <div class="mb-3">
                            <label for="userConfirmPassword" class="form-label">Confirm Password</label>
                            <input type="password" class="form-control" id="userConfirmPassword" name="confirm_password" required>
                        </div>
                        <div class="mb-3 form-check">
                            <input type="checkbox" class="form-check-input" id="userTerms" required>
                            <label class="form-check-label" for="userTerms">I agree to the Terms and Conditions</label>
                        </div>
                        <button type="submit" class="btn btn-primary w-100">Register</button>
                    </form>
                </div>
                
                <!-- Doctor Registration Form -->
                <div class="tab-pane fade" id="doctor-register" role="tabpanel" aria-labelledby="doctor-reg-tab">
                    <form action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]); ?>" method="post">
                        <input type="hidden" name="role" value="doctor">
                        <div class="row mb-3">
                            <div class="col">
                                <label for="doctorFirstName" class="form-label">First Name</label>
                                <input type="text" class="form-control" id="doctorFirstName" name="first_name" required>
                            </div>
                            <div class="col">
                                <label for="doctorLastName" class="form-label">Last Name</label>
                                <input type="text" class="form-control" id="doctorLastName" name="last_name" required>
                            </div>
                        </div>
                        <div class="mb-3">
                            <label for="doctorEmail" class="form-label">Email address</label>
                            <input type="email" class="form-control" id="doctorEmail" name="email" required>
                        </div>
                        <div class="mb-3">
                            <label for="doctorPassword" class="form-label">Password</label>
                            <input type="password" class="form-control" id="doctorPassword" name="password" required>
                            <div class="form-text">Password must be at least 6 characters long.</div>
                        </div>
                        <div class="mb-3">
                            <label for="doctorConfirmPassword" class="form-label">Confirm Password</label>
                            <input type="password" class="form-control" id="doctorConfirmPassword" name="confirm_password" required>
                        </div>
                        <div class="mb-3 form-check">
                            <input type="checkbox" class="form-check-input" id="doctorTerms" required>
                            <label class="form-check-label" for="doctorTerms">I agree to the Terms and Conditions</label>
                        </div>
                        <button type="submit" class="btn btn-primary w-100">Register</button>
                    </form>
                </div>
            </div>
            
            <?php endif; ?>
            
            <div class="mt-4 text-center">
                <p>Already have an account? <a href="login.php">Login here</a></p>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html> 