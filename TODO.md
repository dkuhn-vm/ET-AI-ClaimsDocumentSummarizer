- [x] test text summarization
- [x] create reusable function for summarization
- [x] get document
- [x] check file type of document
- [x] if pdf, split and get each page
    - [ ] convert to images.
    - [ ] use pytesseract to get text from image
    - [ ] summarize each page


Final Trend Summary: ## High-Level Trend Summary for Development & QA Leaders:

**Overall:**

- The provided reports highlight numerous ongoing incidents impacting production environments across various systems and applications.
- Critical alerts and performance degradation issues are prevalent, affecting services like Receivables Hub, Zenoss server, and ePayments system.
- Root causes remain unknown for several incidents, hindering effective resolution.

**Key Trends:**

* **Infrastructure & Hardware:**
    - Connectivity issues and server outages affecting storage operations and production environments.
    - Critical alerts on Linux and Windows servers associated with Nasuni filer suggest potential storage-related problems.
* **Applications & Software:**
    - Application crashes, errors, and performance degradation impacting ePayments system, PolicyWriter/PolicyPro, and Navigator application.
    - Splunk reporting a "Critical" error code related to the ePayments system.
* **Alerting & Monitoring:**
    - Many incidents lack clear alert criteria or specific impact details, hindering comprehensive understanding and prioritization.
* **Communication & Resolution:**
    - Lack of detailed resolution steps in several reports creates uncertainty about ongoing efforts and planned actions.

**Recommendations:**

* **Enhanced Monitoring:** Implement granular monitoring of infrastructure, applications, and services to capture relevant data for future incidents.
* **Clear Alerting:** Define specific and actionable alert criteria to facilitate timely response and mitigation of critical issues.
* **Detailed Reporting:** Provide comprehensive incident summaries with clear cause, impact, resolution steps, and updates for improved transparency and accountability.
* **Improved Communication:** Regularly share incident updates and resolution plans with relevant stakeholders to enhance collaboration and ensure business continuity.