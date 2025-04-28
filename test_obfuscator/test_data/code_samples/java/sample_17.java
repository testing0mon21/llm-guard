import java.util.ArrayList;
import java.util.List;

public class QuickSort {
    /**
     * Quicksort implementation for integers.
     */
    public static List<Integer> quickSort(List<Integer> arr) {
        if (arr.size() <= 1) {
            return arr;
        }
        
        Integer pivot = arr.get(arr.size() / 2);
        List<Integer> left = new ArrayList<>();
        List<Integer> middle = new ArrayList<>();
        List<Integer> right = new ArrayList<>();
        
        for (Integer x : arr) {
            if (x < pivot) {
                left.add(x);
            } else if (x.equals(pivot)) {
                middle.add(x);
            } else {
                right.add(x);
            }
        }
        
        List<Integer> result = new ArrayList<>();
        result.addAll(quickSort(left));
        result.addAll(middle);
        result.addAll(quickSort(right));
        
        return result;
    }
}