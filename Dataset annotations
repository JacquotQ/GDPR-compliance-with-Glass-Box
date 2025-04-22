import os
import json
import re
import requests
import time
import sys
from tqdm import tqdm

# 配置信息
MAX_RETRIES = 3
RETRY_DELAY = 5  # 秒

# 设置DeepSeek API Key
DEEPSEEK_API_KEY = ""  # 请替换为你的DeepSeek API密钥
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# 定义标注的 Prompt 模板
PROMPT_TEMPLATE = """
Analyze the following GDPR case text, one file is one case, read through the whole case and extract information strictly following these requirements:
Extract structured data from the case, ensuring each field contains only the exact values from the case text. You read the case, then analyze the case based on the following requirements, you can only have one answer based on each requirement.Your ouput should be one line json data.

For fields that could have multiple valid values, extract all applicable values, but keep it concise and in the same line.

Requirements:
1. For each of the following data category components, output a separate field with a binary value (1 indicates the information is present in the text; 0 indicates it is absent):
- data_category_Basic_personal_data: Set to 1 if the case about basic personal data "address", "DoB", or "ID"; otherwise, 0.
- data_category_Special_category_data: Set to 1 if the text contains any of "race", "religion", "sexual orientation", "biometrics", or "genetic data"; otherwise, 0.
- data_category_Criminal_data: Set to 1 if the case about "criminal records" or "court judgments"; otherwise, 0.
- data_category_Financial_location_data: Set to 1 if the case about "financial" or "location" data; otherwise, 0.
- data_category_Children_data: Set to 1 if the case "children", "child" related, or "Children's data"; otherwise, 0.

2. For each of the following processing basis components, output a separate field with a binary value (1 for presence, 0 for absence) based strictly on the text:
- data_processing_basis_Legitimate_interest: Set to 1 if the processing based on "Legitimate interest" or related descriptions (e.g., "marketing"); otherwise, 0.
- data_processing_basis_contract_performance: Set to 1 if the data processing is required by the contract; otherwise, 0.
- data_processing_basis_Consent: Set to 1 if the data processing based on "Consent"; otherwise, 0.
- data_processing_basis_Legal_obligation: Set to 1 if the processing required by "Legal obligation" or related descriptions (e.g., "tax reporting"); otherwise, 0.
- data_processing_basis_Protection_of_vital_interests: Set to 1 if the processing is to "Protection of vital interests"; otherwise, 0.
- data_processing_basis_Performance_of_public_task: Set to 1 if the processing is for "Performance of public task"; otherwise, 0.

3. fine_amount:
   - Extract exact amount with currency (e.g., "200,000")
   - "Not specified" if unclear
   - 0 if no fine

4. country:
   - Full country name in English (e.g., "Germany")

5. company_industry:
   - choose from: Public sector(court, government, or company provide service for the government); Marketing(advertisement, profiling, case about marketing)
Eduction(any school); Medical(hospital and medical investigation, medical related);Retail(everything about selling product and service, and if unspecific)
Human resources(this case about HR department in any kind of company);Security Service(monitor, servelience);Leisure(entertainment, fitness)
Social Media(any internet company, website, online market, social media, etc);Individual; Insurance

6. gdpr_clause:
   - Specific article reference (e.g., "Article 9(2)(a)")

7. gdpr_conflict: choose below
   - yes
   - No conflict (if not mention)

8. For each of the following violation nature components, output a separate field with a binary value (1 for presence, 0 for absence) based strictly on the text:
- violation_nature_Breach_of_Data_processing_principle: Set to 1 if the text contains "Breach of Data processing principle" or related descriptions (e.g., "Data minimization"); otherwise, 0.
- violation_nature_Violation_of_data_subject_rights: Set to 1 if the text contains "Violation of data subject rights"; otherwise, 0.
- violation_nature_Breach_of_data_security: Set to 1 if the text contains "Breach of data security"; otherwise, 0.
- violation_nature_Violation_of_Data_processing_obligation: Set to 1 if the text contains "Violation of Data processing obligation" or related descriptions (e.g., "DPO"); otherwise, 0.

9. 1 if applicable, 0 if unapplicable
-free_speech_exception
-country_security_exception
-Criminal_investigation_exception

10. violation result:
- violation_result: if violate, 1, not 0

11. Affected_data_volume:
   - Numerical value or qualitative estimation (e.g., "500,000 records affected").
   - unspecific

12. Date:
   - Year of this case

Rules:
- Values MUST come directly from text.
- No assumptions or interpretations.
- Maintain original phrasing for extracted values.
- If there are multiple valid values, you can output them as an array (e.g., ["value1", "value2"]).
- You can only return one line JSON format.

Text:
{text}


Return valid JSON in this structure:
"""

def extract_json_from_response(response):
    """Extract JSON content from model response"""
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"error": "No valid JSON found"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {e}"}

def annotate_text(text):
    """Call DeepSeek API for GDPR data annotation with retry mechanism"""
    raw_output = ""
    retry_count = 0
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    while retry_count < MAX_RETRIES:
        try:
            print(f"准备调用DeepSeek API...（尝试 {retry_count+1}/{MAX_RETRIES}）")
            print(f"文本长度: {len(text)} 字符")
                
            # 记录开始时间
            start_time = time.time()
            
            # 构造API请求数据
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a GDPR compliance analyst."},
                    {"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}
                ],
                "temperature": 0.3,
                "max_tokens": 1024
            }
            
            print("发送请求到DeepSeek API...")
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=120)
            
            # 记录结束时间
            elapsed_time = time.time() - start_time
            print(f"DeepSeek API调用完成，耗时: {elapsed_time:.2f}秒")

            # 检查API响应
            if response.status_code == 200:
                response_data = response.json()
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    raw_output = response_data["choices"][0]["message"]["content"]
                    print(f"提取到文本响应，长度: {len(raw_output)}")
                    
                    if raw_output:
                        result = extract_json_from_response(raw_output)
                        print(f"JSON提取结果: {'成功' if 'error' not in result else '失败: ' + result.get('error', '')}")
                        return result
                    else:
                        print("警告: API返回了空响应")
                        return {"error": "Empty text in API response", "raw_output": ""}
                else:
                    print("API响应格式不正确")
                    print(f"响应内容: {response_data}")
                    return {"error": "Incorrect response format from API", "raw_output": str(response_data)}
            else:
                print(f"API请求失败，状态码: {response.status_code}")
                print(f"响应内容: {response.text}")
                return {"error": f"API request failed with status code {response.status_code}", "raw_output": response.text}
                
        except requests.exceptions.Timeout:
            print("API请求超时")
            retry_count += 1
            if retry_count < MAX_RETRIES:
                print(f"等待 {RETRY_DELAY} 秒后重试...")
                time.sleep(RETRY_DELAY)
            continue
                
        except requests.exceptions.RequestException as e:
            print(f"API请求错误: {e}")
            retry_count += 1
            wait_time = min(RETRY_DELAY * (2 ** retry_count), 60)  # 指数退避策略，最长等待60秒
            if retry_count < MAX_RETRIES:
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            continue
                
        except Exception as e:
            print(f"调用API时发生未预期的错误: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Unexpected error: {str(e)}", "raw_output": ""}
            
    # 所有重试都失败后的返回
    return {"error": "Max retries exceeded", "raw_output": raw_output}

def extract_number_from_filename(filename):
    """从文件名中提取数字编号"""
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0  # 如果没有找到数字，返回0作为默认值

def process_cases(input_dir):
    """Process all GDPR case files in a directory"""
    if not os.path.isdir(input_dir):
        print("Error: Input path is not a directory or does not exist.")
        return

    # 获取所有txt文件
    files = []
    for root, _, filenames in os.walk(input_dir):
        for file in filenames:
            if file.endswith('.txt'):
                files.append(os.path.join(root, file))
    
    # 根据文件名中的数字编号进行排序
    files.sort(key=lambda x: extract_number_from_filename(os.path.basename(x)))
    
    # 处理排序后的文件
    for file_path in tqdm(files, desc="Processing files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            annotations = annotate_text(content)
            
            output_file = f"{os.path.splitext(file_path)[0]}.json"
            result = {
                "metadata": {
                    "source_file": os.path.basename(file_path),
                    "text_length": len(content)
                },
                "annotations": annotations
            }
            
            # Save annotation results to JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"Processed: {file_path} -> {output_file}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def test_api_connection():
    """测试API连接是否正常工作"""
    print("测试DeepSeek API连接...")
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": "Hello, are you working?"}
            ],
            "max_tokens": 10
        }
        
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            print("API连接测试成功!")
            return True
        else:
            print(f"API连接测试失败，状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False
    except Exception as e:
        print(f"API连接测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    input_path = ""  # 修改为你的输入目录路径
    
    # 确保设置了DeepSeek API密钥
    if DEEPSEEK_API_KEY == "your-deepseek-api-key":
        print("错误: 请先在代码中设置你的DeepSeek API密钥!")
        exit(1)
    
    # 显示运行环境信息
    print("Python版本:", sys.version)
    
    # 测试API连接
    if not test_api_connection():
        print("错误: API连接测试失败，请检查API密钥和网络连接")
        exit(1)
    
    # 运行主程序
    print(f"开始处理目录: {input_path}")
    process_cases(input_path)
    
