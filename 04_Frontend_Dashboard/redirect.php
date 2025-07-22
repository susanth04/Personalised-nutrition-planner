<?php
require_once 'config/session.php';

// Ensure the user is logged in
requireLogin();

// Get the user's role
$role = $_SESSION['user_role'];

// Redirect based on role
if ($role === 'doctor') {
    // Redirect to doctor dashboard
    header("Location: http://localhost:3000/digital-twin");
    exit;
} else {
    // Redirect to user dashboard
    header("Location: http://localhost:3000/meal-plan");
    exit;
}
?> 