<?php
require_once 'config/session.php';
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unauthorized - Nutrition Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f5f8fa;
            font-family: Arial, sans-serif;
            padding-top: 100px;
        }
        .error-container {
            max-width: 600px;
            margin: 0 auto;
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
            padding: 40px;
            text-align: center;
        }
        .error-icon {
            font-size: 80px;
            color: #dc3545;
            margin-bottom: 20px;
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
        }
        .btn-primary {
            background-color: #4CAF50;
            border-color: #4CAF50;
            margin-top: 20px;
        }
        .btn-primary:hover {
            background-color: #3e8e41;
            border-color: #3e8e41;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="error-container">
            <div class="error-icon">🔒</div>
            <h1>Access Denied</h1>
            <p class="lead">You don't have permission to access this page.</p>
            
            <?php if (isLoggedIn()): ?>
                <p>You are logged in as: <strong><?php echo htmlspecialchars($_SESSION['user_email']); ?></strong> (<?php echo ucfirst($_SESSION['user_role']); ?>)</p>
                <p>Please navigate to your appropriate dashboard or contact support if you believe this is an error.</p>
                <div class="mt-4">
                    <a href="redirect.php" class="btn btn-primary">Go to Dashboard</a>
                    <a href="logout.php" class="btn btn-outline-secondary ms-2">Logout</a>
                </div>
            <?php else: ?>
                <p>Please log in to access your dashboard.</p>
                <a href="login.php" class="btn btn-primary">Login</a>
            <?php endif; ?>
        </div>
    </div>
</body>
</html> 