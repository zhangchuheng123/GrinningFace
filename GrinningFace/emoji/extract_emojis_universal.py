#!/usr/bin/env python3
"""
Universal emoji extractor: Extract emoji characters from meta_relabel_dedup_xxx.json files and generate xxx.txt files.
Processes ALL matching files in the directory.
"""

import json
import os
import glob
from pathlib import Path

def extract_emojis_from_json(json_file_path):
    """Extract emoji characters from JSON file."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract all 'char' fields (emoji characters)
        emojis = [item['char'] for item in data if 'char' in item]
        return emojis
    except Exception as e:
        print(f"Error reading {json_file_path}: {e}")
        return []

def write_emojis_to_file(emojis, output_file_path):
    """Write emojis to a text file, all in one line."""
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(''.join(emojis))
        print(f"✅ Written {len(emojis)} emojis to {output_file_path}")
        return True
    except Exception as e:
        print(f"❌ Error writing to {output_file_path}: {e}")
        return False

def main():
    current_dir = Path(__file__).parent
    
    # Find all meta_relabel_dedup_*.json files
    json_pattern = str(current_dir / "meta_relabel_dedup_*.json")
    json_files = sorted(glob.glob(json_pattern))  # Sort for consistent order
    
    if not json_files:
        print(f"❌ No meta_relabel_dedup_*.json files found in {current_dir}")
        print(f"📁 Looking for pattern: {json_pattern}")
        return
    
    print(f"📁 Found {len(json_files)} JSON files to process:")
    for json_file in json_files:
        print(f"   - {Path(json_file).name}")
    
    processed_count = 0
    successful_count = 0
    
    for json_file_path in json_files:
        json_file = Path(json_file_path)
        print(f"\n📖 Processing {json_file.name}...")
        
        try:
            # Extract the xxx part from meta_relabel_dedup_xxx.json
            filename = json_file.stem  # removes .json extension
            xxx_part = filename.replace('meta_relabel_dedup_', '')
            
            if not xxx_part:
                print(f"⚠️  Could not extract suffix from {filename}")
                continue
            
            # Extract emojis
            emojis = extract_emojis_from_json(json_file_path)
            if not emojis:
                print(f"⚠️  No emojis found in {json_file.name}")
                continue
                
            print(f"📊 Found {len(emojis)} emojis")
            
            # Generate xxx.txt file
            output_file = current_dir / f"pure_{xxx_part}.txt"
            success = write_emojis_to_file(emojis, output_file)
            
            if success:
                successful_count += 1
                # Show a preview
                if emojis:
                    preview = ''.join(emojis[:10]) + ('...' if len(emojis) > 10 else '')
                    print(f"👀 Preview: {preview}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")
            continue
    
    print(f"\n🎉 Processing complete!")
    print(f"📊 Processed: {processed_count}/{len(json_files)} files")
    print(f"✅ Successful: {successful_count}/{processed_count} conversions")
    
    # List all generated txt files
    txt_files = sorted(glob.glob(str(current_dir / "*.txt")))
    if txt_files:
        print(f"\n📋 All text files in directory:")
        for txt_file in txt_files:
            txt_path = Path(txt_file)
            try:
                file_size = txt_path.stat().st_size
                print(f"   - {txt_path.name} ({file_size} bytes)")
            except:
                print(f"   - {txt_path.name} (size unknown)")
    else:
        print(f"\n⚠️  No text files found in directory.")

if __name__ == "__main__":
    main()