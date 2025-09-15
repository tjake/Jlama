package com.github.tjake.jlama.net;

import org.junit.jupiter.api.Test;

import static com.github.tjake.jlama.net.grpc.JlamaService.isPowerOfTwoUsingBitwiseOperation;
import static com.github.tjake.jlama.net.grpc.JlamaService.nextPowerOfTwo;
import static org.junit.jupiter.api.Assertions.*;


public class JlamaServiceUnitTest {

    @Test
    public void nextPowerOfTwoTest(){
        assertEquals(2, nextPowerOfTwo(2));
        assertEquals(4, nextPowerOfTwo(3));
        assertEquals(256, nextPowerOfTwo(256));
        assertEquals(512, nextPowerOfTwo(257));
    }

    @Test
    public void isPowerOfTwoTest(){
        assertTrue(isPowerOfTwoUsingBitwiseOperation(2));
        assertFalse(isPowerOfTwoUsingBitwiseOperation(3));
        assertFalse(isPowerOfTwoUsingBitwiseOperation(-1));
        assertFalse(isPowerOfTwoUsingBitwiseOperation(-2));
    }
}
