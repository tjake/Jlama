package com.github.tjake.jlama.net;

import com.github.tjake.jlama.net.grpc.JlamaService;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SimpleTest {

    @Test
    public void powerOfTwoTest(){
        assertEquals(2, JlamaService.nextPowerOfTwo(2));
        assertEquals(4, JlamaService.nextPowerOfTwo(3));
        assertEquals(256, JlamaService.nextPowerOfTwo(256));
        assertEquals(512, JlamaService.nextPowerOfTwo(257));
    }
}
