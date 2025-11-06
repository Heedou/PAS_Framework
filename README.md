# PAS_Framework

## Contributors

<table>
	<tr>
		<th>Name</th>		
		<th>Affiliation</th>
		<th>Email</th>
	</tr>
  <tr>
		<td>Sangyub Lee</td>		
		<td>Intelligence and Informatics Lab,<br>Korea University, Seoul, South Korea & Korea National Police Agency</td>
		<td>yubii2@korea.ac.kr</td>
	</tr>
	<tr>
		<td>Heedou Kim</td>		
		<td>Data Mining and Information Systems Lab,<br>Korea University, Seoul, South Korea</td>
		<td>heedou123@korea.ac.kr</td>
	</tr>
	<tr>
		<td>Hyuncheol Kim*</td>		
		<td>Intelligence and Informatics Lab,<br>Korea University, Seoul, South Korea</td>
		<td>harrykim@korea.ac.kr</td>
	</tr>
</table>

- &ast;: *Corresponding Author*

# PAS: Police Action Scenario Framework

<img width="500" height="300" alt="FIg1-PAS Construction with Example" src="https://github.com/user-attachments/assets/18b5e23d-70d8-4640-a7fb-e5b6c58f196b" />


**PAS (Police Action Scenario)** is a dedicated framework for evaluating Large Language Models (LLMs) in real-world policing contexts.

Modern policing requires nuanced judgment and situational awareness—standard benchmarks alone are not sufficient. PAS introduces a scenario-based, multi-stage evaluation method designed specifically for policing tasks.

## 🔍 Evaluation Pipeline

PAS defines LLM evaluation as a five-stage process:

- **S: Police Action Scenarios**  
  Situation-driven tasks reflecting real-world policing needs.

- **R: Reference Responses**  
  Expert-crafted gold answers created with input from law enforcement professionals.

- **G: Response Generation**  
  LLM-generated outputs based on the given scenarios.

- **M: Core Evaluation Metrics**  
  Task-relevant metrics and evaluation methodologies tailored for public safety applications.

- **P: Policing LLM Performance Evaluation**  
  Final assessment of the LLM’s effectiveness, accuracy, and fitness for deployment in policing.

Formally expressed as:  
`E_police = f(S, R, G, M, P)`

---

> PAS fills the gap in evaluating LLMs for law enforcement by combining structured scenarios, expert benchmarks, and targeted metrics.
