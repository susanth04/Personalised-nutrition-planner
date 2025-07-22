<?php
require_once 'config/session.php';

// If logged in, redirect to the dashboard, otherwise to login
if (isLoggedIn()) {
    header("Location: redirect.php");
} else {
    header("Location: login.php");
}
exit;
?> 