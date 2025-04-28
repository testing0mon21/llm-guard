public class EmailValidator {
    /**
     * Validate an email address using regex.
     */
    public static boolean validateEmail(String email) {
        String pattern = "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$";
        return email.matches(pattern);
    }
}