1. URL Structure Features
URL Length: Longer URLs may indicate phishing attempts.
Number of Special Characters: Count characters like -, _, @, ?, =, &, %, and .; phishing URLs often use special characters to obscure true destinations.
Number of Subdomains: Phishing URLs often use multiple subdomains to appear legitimate.
Presence of IP Address: If the URL contains an IP address instead of a domain name, it may indicate phishing.
HTTPS Usage: The presence of "https" can be a positive indicator, though it's not foolproof as phishing sites sometimes use HTTPS.

2. Domain-Based Features
Domain Age: Newly registered domains may be used for phishing. Use a WHOIS lookup to determine the age.
Domain Reputation: Check domain reputation through databases or APIs (e.g., VirusTotal or Google Safe Browsing).
Top-Level Domain (TLD): Some phishing sites use unusual or cheap TLDs, such as .tk or .cf.
Registrar: Some registrars have a history of being lenient on phishing sites. Analyzing the registrar can provide insights.

3. Content-Based Features
Suspicious Keywords: Look for phishing-indicative words in the URL, such as "login", "verify", "account", "update", or "secure".
Presence of Brand Names: Phishing URLs often include well-known brand names to deceive users (e.g., paypal-login.com).
Keyword Frequency: Count occurrences of specific keywords that might be commonly used in phishing URLs.

4. Entropy and Randomness-Based Features
Entropy of URL: Phishing URLs often have a higher entropy score due to a seemingly random string of characters.
Domain and Path Entropy: Calculate entropy specifically for the domain or path part of the URL, as phishing sites often have random-looking paths.

5. HTML and JavaScript Features (For Page-Level Analysis)
Number of External Links: A high number of external links on a page may be suspicious, especially if they lead to different domains.
Redirects: Phishing pages often use JavaScript or meta-refresh to redirect users to different pages.
Presence of Suspicious Forms: Look for form elements where action attributes link to external or unusual URLs.
Obfuscation: Many phishing pages use obfuscated JavaScript code to hide their intent.

6. Statistical & Frequency Features
Frequency of Occurrences: Certain suspicious characters or patterns can be more frequent in phishing URLs.
Path Length and Query Parameter Count: The length of the path and the number of query parameters can also provide clues.

7. Behavioral and Temporal Features (Optional)
Access Patterns: Analyze how frequently URLs are accessed or flagged by users, though this requires access to dynamic usage data.
Time of Day/Registration Time: Newly registered or recently active domains during non-standard hours may be suspicious.

8. PageRank and Link Analysis (For Large-Scale Detection)
PageRank Score: Phishing pages often have a lower PageRank, indicating a lack of legitimate backlinks.
Inbound Link Analysis: Legitimate sites usually have inbound links from reputable domains, whereas phishing pages might lack these.