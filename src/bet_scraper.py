from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
from datetime import datetime

def scroll_to_element(driver, element):
    """Scroll element into view using JavaScript"""
    driver.execute_script("arguments[0].scrollIntoView(true);", element)
    time.sleep(1)  #

try:
    cricket_data = {
        'timestamp': datetime.now().isoformat(),
        'formats': {}
    }
    driver = webdriver.Chrome()
    driver.get("https://sportsbook.draftkings.com/sports/cricket")
    wait = WebDriverWait(driver, 10)

    # Get all match format types
    match_types = wait.until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, "league-link__link-name"))
    )
    
    # Extract format names
    format_names = [format_type.text for format_type in match_types]
    print(f"Found formats: {format_names}")

    # Navigate through each format
    for format_name in format_names:
        try:
            print(f"\nProcessing {format_name} matches...")
            cricket_data['formats'][format_name] = {
                'matches': {}
            }
            # Click on the format
            format_link = wait.until(
                EC.element_to_be_clickable((By.LINK_TEXT, format_name))
            )
            scroll_to_element(driver,format_link)
            format_link.click()
            
            # Wait for the matches to load
            time.sleep(2)  # Consider replacing with explicit wait

            matches_elements = wait.until(
                EC.presence_of_all_elements_located((By.CLASS_NAME,"sportsbook-event-accordion__title"))
            )
            match_names = [match_element.text for match_element in matches_elements]
            print (match_names)
            for match_name in match_names:
                print(f"\nProcessing {match_name} match...")
                
                #get the date if match is not live
                if '\n' in match_name:
                    match_name_first_team  = match_name.split('\n')[0]
                    print(f"HERE IS {match_name_first_team}")
                    match_date_xpath = f"//a[@class='sportsbook-event-accordion__title'][.//div[contains(text(),'{match_name_first_team}')]]/following-sibling::span[@class='sportsbook-event-accordion__date']"
                    match_name_xpath = f"//a[@class='sportsbook-event-accordion__title'][.//div[contains(text(),'{match_name_first_team}')]]"
                    try:
                        match_date_element = wait.until(
                            EC.presence_of_element_located((By.XPATH, match_date_xpath))
                        )
                        match_date = match_date_element.text
                    except Exception as e:
                        print(f"Could not find date for {match_name}: {e}")
                else:
                    match_date = "Live"
                    match_name_xpath = f"//a[@class='sportsbook-event-accordion__title'][text()='{match_name}']"

                print (f"Match Date is {match_date}")

                cricket_data['formats'][format_name]['matches'][match_name] = {
                    'date': match_date,
                    'betting_categories': {}
                }
                
                match_link = wait.until(
                    EC.element_to_be_clickable((By.XPATH, match_name_xpath))
                )
                scroll_to_element(driver, match_link)
                match_link.click()

                bet_category_elements = wait.until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, "tab-switcher-sub-tab-child"))
                )
                
                # Extract format names
                bet_categories = [bet_category_element.text for bet_category_element in bet_category_elements]
                print(f"Found bet categories: {bet_categories}")
                # Navigate through each format
                for bet_category in bet_categories:
                    cricket_data['formats'][format_name]['matches'][match_name]['betting_categories'][bet_category] = {
                        'bet_types': {}
                    }

                    bet_category_link = wait.until(
                        EC.element_to_be_clickable((By.LINK_TEXT, bet_category))
                    )
                    scroll_to_element(driver, bet_category_link)
                    bet_category_link.click()

                    bet_type_elements = wait.until(
                        EC.presence_of_all_elements_located((By.CLASS_NAME, "cb-collapsible-header"))
                    )
                    
                    # Extract bet type names
                    bet_types = [bet_type_element.text for bet_type_element in bet_type_elements]
                    print(f"Found bet types: {bet_types}")
                    bet_type_info = {}
                    for bet_type in bet_types:
                        try:
                            # Find the parent div that contains this bet type header
                            parent_div_xpath = f"//div[contains(@class, 'cb-collapsible') and .//h2[contains(@class, 'cb-collapsible-header') and contains(text(), '{bet_type}')]]"
                            parent_div = wait.until(
                                EC.presence_of_element_located((By.XPATH, parent_div_xpath))
                            )
                            
                            bet_info = {
                                'description': None,
                                'betting_options': []
                            }
                            # Check if this div contains the p element with the specific class
                            try:
                                label_element = parent_div.find_element(By.CSS_SELECTOR, "p.cb-market__label--truncate-strings")
                                bet_description = label_element.text
                                print(f"Bet type: {bet_type} has label: {bet_description}")
                            except:
                                print(f"Bet type: {bet_type} has no label")
                                bet_description = ""
                            
                            bet_info['description'] = bet_description

                            betting_buttons = parent_div.find_elements(By.CSS_SELECTOR, "button.cb-market__button.cb-market__button--regular")
        
                            for button in betting_buttons:
                                try:
                                    # Get the option name and odds
                                    option_name = button.find_element(By.CSS_SELECTOR, "span.cb-market__button-title").text
                                    try:
                                        points = button.find_element(By.CSS_SELECTOR, "span.cb-market__button-points").text
                                        option_name = f"{option_name} {points}"  # Combine name with points
                                    except:
                                        pass  # No points span found, just use the base option name

                                    option_odds = button.find_element(By.CSS_SELECTOR, "span.cb-market__button-odds").text
                                    
                                    print (f"bet option: {option_name} has odds {option_odds}")
                                    
                                    bet_info['betting_options'].append({
                                        'option': option_name,
                                        'odds': option_odds
                                    })

                                except Exception as e:
                                    print(f"Error processing betting option: {e}")
                                    continue
                            
                            bet_type_info[bet_type] = bet_info
                
                        except Exception as e:
                            print(f"Error processing bet type {bet_type}: {e}")
                            continue
                    
                    cricket_data['formats'][format_name]['matches'][match_name]['betting_categories'][bet_category]['bet_types'] = bet_type_info

                    time.sleep(2)

                driver.back()
                        # bet_infos = wait.until(
                        #     EC.presence_of_all_elements_located((By.CLASS_NAME,"sportsbook-outcome-cell__label"))
                        # )
                        # # print("working till here")
                        # bet_odds_elements = wait.until(
                        #     EC.presence_of_all_elements_located((By.CLASS_NAME,"sportsbook-odds.american.default-color"))
                        # ) 


            # # Here you can add code to collect match data, for example:
            # bet_categories_elements = wait.until(
            #     EC.presence_of_all_elements_located((By.CLASS_NAME,"sportsbook-category-tab-name"))
            # )
            # bet_category_names = [bet_category.text for bet_category in bet_categories_elements]

            # #go to match lines
            # bet_category_link = wait.until(
            #         EC.element_to_be_clickable((By.LINK_TEXT, bet_category_names[0]))
            #     )
            # bet_category_link.click()

            # #go to money line
            # bet_type_elements = wait.until(
            #         EC.presence_of_all_elements_located((By.CLASS_NAME,"sportsbook-tabbed-subheader__tab"))
            #     )
            # bet_type_names = [bet_type.text for bet_type in bet_type_elements]
            # bet_type_link = wait.until(
            #             EC.element_to_be_clickable((By.LINK_TEXT, bet_type_names[0]))
            #         )
            # bet_type_link.click()



            # for bet_category_name in bet_category_names:
            #     print(f"\nProcessing {bet_category_name} category...")
            #     bet_category_link = wait.until(
            #         EC.element_to_be_clickable((By.LINK_TEXT, bet_category_name))
            #     )
            #     bet_category_link.click()
                # bet_type_elements = wait.until(
                #     EC.presence_of_all_elements_located((By.CLASS_NAME,"sportsbook-tabbed-subheader__tab"))
                # )
                # bet_type_names = [bet_type.text for bet_type in bet_type_elements]

                # for bet_type_name in bet_type_names:
                #     print(f"\nProcessing {bet_type_name} type...")
                #     bet_type_link = wait.until(
                #         EC.element_to_be_clickable((By.LINK_TEXT, bet_type_name))
                #     )
                #     bet_type_link.click()



                    # matches_elements = wait.until(
                    #     EC.presence_of_all_elements_located((By.CLASS_NAME,"sportsbook-event-accordion__title"))
                    # )
                    # match_names = [match_element.text for match_element in matches_elements]

                    # for match in match_names:
                    #     print(f"\nProcessing {match} match...")
                    #     bet_infos = wait.until(
                    #         EC.presence_of_all_elements_located((By.CLASS_NAME,"sportsbook-outcome-cell__label"))
                    #     )
                    #     # print("working till here")
                    #     bet_odds_elements = wait.until(
                    #         EC.presence_of_all_elements_located((By.CLASS_NAME,"sportsbook-odds.american.default-color"))
                    #     )

                    #     print("working till here")
                    #     bet_winner_conditions = [bet_info.text for bet_info in bet_infos]

                    #     # for bet_winner_condition in bet_winner_conditions:
                    #     #     print(f"\nOdds for {bet_winner_condition}")
                    #     bet_odds = [bet_odds_element.text for bet_odds_element in bet_odds_elements]
                    #     for bet_winner_condition,bet_odd in zip(bet_winner_conditions, bet_odds):
                    #         print (f"\nOdds for {bet_winner_condition} are {bet_odd}"
            
            # Navigate back to main cricket page
            driver.get("https://sportsbook.draftkings.com/sports/cricket")
            time.sleep(2)  # Allow page to load
            
        except Exception as e:
            print(f"Error processing {format_name}: {e}")
            driver.get("https://sportsbook.draftkings.com/sports/cricket")
            time.sleep(2)
            continue

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    driver.quit()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'cricket_betting_data_{timestamp}.json'

with open(filename, 'w', encoding='utf-8') as f:
    json.dump(cricket_data, f, indent=2, ensure_ascii=False)