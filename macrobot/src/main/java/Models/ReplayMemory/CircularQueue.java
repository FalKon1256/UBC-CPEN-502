package Models.ReplayMemory;

import java.util.LinkedList;

/** This class implements a replay memory for any type T.
 *  The capacity of the memory must be specified upon construction.
 *  The memory will discard the oldest items that do not fit.
 */


public class CircularQueue<T> extends LinkedList<T> {
    private int capacity = 10;

    public CircularQueue(int capacity){
        this.capacity = capacity;
    }

    public boolean add(T e) {
        if(size() >= capacity)
            removeFirst();
        return super.add(e);
    }

    public Object [] toArray() {
        return super.toArray();
    }
}