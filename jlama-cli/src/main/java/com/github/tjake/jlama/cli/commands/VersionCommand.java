/*
 * Copyright 2024 T Jake Luciani
 *
 * The Jlama Project licenses this file to you under the Apache License,
 * version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.github.tjake.jlama.cli.commands;

import java.util.Properties;
import java.io.InputStream;
import com.github.tjake.jlama.cli.JlamaCli;

import picocli.CommandLine;

@CommandLine.Command(name = "version", description = "Display JLama version information", abbreviateSynopsis = true)
public class VersionCommand extends JlamaCli {
    @Override
    public void run() {
        try (InputStream is = getClass().getResourceAsStream("/META-INF/maven/com.github.tjake/jlama-cli/pom.properties")) {
            if (is != null) {
                Properties props = new Properties();
                props.load(is);
                System.out.println("JLama version " + props.getProperty("version"));
            } else {
                System.out.println("Version information not available");
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }
}
