<?php
// Start session if not already started
if (session_status() === PHP_SESSION_NONE) {
    session_start();
}

// Check if user is logged in
function isLoggedIn() {
    return isset($_SESSION['user_id']);
}

// Check if user is a doctor
function isDoctor() {
    return isset($_SESSION['user_role']) && $_SESSION['user_role'] === 'doctor';
}

// Check if user is a patient/regular user
function isUser() {
    return isset($_SESSION['user_role']) && $_SESSION['user_role'] === 'user';
}

// Redirect if not logged in
function requireLogin() {
    if (!isLoggedIn()) {
        header("Location: login.php");
        exit;
    }
}

// Redirect if not a doctor
function requireDoctor() {
    requireLogin();
    if (!isDoctor()) {
        header("Location: unauthorized.php");
        exit;
    }
}

// Redirect if not a user
function requireUser() {
    requireLogin();
    if (!isUser()) {
        header("Location: unauthorized.php");
        exit;
    }
}

// Get current user info
function getCurrentUser() {
    if (isLoggedIn()) {
        return [
            'id' => $_SESSION['user_id'],
            'email' => $_SESSION['user_email'],
            'role' => $_SESSION['user_role'],
            'first_name' => $_SESSION['user_first_name'],
            'last_name' => $_SESSION['user_last_name']
        ];
    }
    return null;
}
?> 