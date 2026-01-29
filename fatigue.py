import datetime
import math
import pandas as pd

class UPSFatigueModel:
    def __init__(self):
        # Your specific rules
        self.rules = {
            "dom_report": 1.0,        # 1:00 prior
            "intl_report_dom": 1.5,   # 1:30 prior (in domicile)
            "dom_pickup": 1.5,        # 1:30 prior (layover)
            "intl_pickup": 2.0,       # 2:00 prior (layover)
            "dom_debrief": 0.25,      # 15 mins
            "intl_debrief": 0.50,     # 30 mins
        }

    def _is_international(self, origin, dest):
        # Simplified logic: If not in this list, assume Intl
        domestic_hubs = ['SDF', 'ONT', 'RFD', 'PHL', 'DFW', 'MIA', 'ANC'] # Add more as needed
        return (origin not in domestic_hubs) or (dest not in domestic_hubs)

    def get_crew_complement(self, block_hours):
        """International Augmentation Rules"""
        if block_hours <= 7.75: return 2  # 7:45
        if block_hours < 12.0: return 3
        return 4

    def calculate_duty(self, flights, is_in_domicile=True):
        if not flights: return None
        
        # 1. Analyze Trip Type
        is_intl = any(self._is_international(f['origin'], f['dest']) for f in flights)
        
        # 2. Calculate Report Time
        if is_in_domicile:
            report_buffer = self.rules['intl_report_dom'] if is_intl else self.rules['dom_report']
        else:
            report_buffer = self.rules['intl_pickup'] if is_intl else self.rules['dom_pickup']
            
        duty_start = flights[0]['dept'] - datetime.timedelta(hours=report_buffer)
        
        # 3. Calculate Debrief Time
        debrief_buffer = self.rules['intl_debrief'] if is_intl else self.rules['dom_debrief']
        duty_end = flights[-1]['arr'] + datetime.timedelta(hours=debrief_buffer)
        
        total_duty = (duty_end - duty_start).total_seconds() / 3600
        
        return {
            "start": duty_start,
            "end": duty_end,
            "duration": total_duty,
            "is_intl": is_intl
        }

    def run_simulation(self, duty_start, duty_end, flights):
        """
        Runs a biomathematical model (SAFTE-style) to estimate effectiveness.
        """
        timeline = []
        current_time = duty_start
        # Starting reservoir (Arbitrary units of 'alertness fuel', max 2880)
        reservoir = 2400 
        
        # Simulation Step size (minutes)
        step = 60 
        
        results = []
        
        # Loop through the duty period hour by hour
        while current_time <= duty_end:
            # 1. Circadian Influence (The WOCL)
            # Body clock low point typically 0300-0500 local domicile time
            hour = current_time.hour
            circadian_penalty = 0
            if 2 <= hour <= 5: 
                circadian_penalty = 15 # Heavy penalty
            elif 13 <= hour <= 15:
                circadian_penalty = 5  # Afternoon dip
            
            # 2. Work Depletion
            # Is this time inside a flight block?
            is_flying = False
            crew_size = 2
            
            for f in flights:
                if f['dept'] <= current_time <= f['arr']:
                    is_flying = True
                    block_len = (f['arr'] - f['dept']).total_seconds()/3600
                    crew_size = self.get_crew_complement(block_len)
                    break
            
            decay_rate = 0
            if is_flying:
                if crew_size == 2: decay_rate = 5
                elif crew_size == 3: decay_rate = 3 # Augmented relief
                elif crew_size == 4: decay_rate = 1.5 # Heavy relief
            else:
                decay_rate = 4 # Ground duty / Preflight
                
            reservoir -= decay_rate
            
            # 3. Calculate Score (0-100 scale)
            # Score = (Reservoir / Max_Reservoir) * 100 - Circadian_Penalty
            raw_score = (reservoir / 2880) * 100
            effectiveness = max(0, min(100, raw_score - circadian_penalty))
            
            risk_color = "🟢"
            if effectiveness < 90: risk_color = "🟡"
            if effectiveness < 85: risk_color = "🟠"
            if effectiveness < 80: risk_color = "🔴"
            if effectiveness < 75: risk_color = "🟣"

            results.append({
                "Time": current_time.strftime("%H:%M"),
                "Effectiveness": round(effectiveness, 1),
                "Risk": risk_color,
                "Activity": "Flight" if is_flying else "Duty"
            })
            
            current_time += datetime.timedelta(minutes=step)
            
        return pd.DataFrame(results)
