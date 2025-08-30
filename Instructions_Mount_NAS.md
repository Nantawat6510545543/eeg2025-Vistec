
# Instruction

## To Open Mount

1. **Remote into server**  
   Connect to `int_crew-s04`

2. **Access the container**  
   - Locate the container named: `4f7df8586b44` (alias: `intern01_moss`)  
   - Right-click on it and select **"Attach Visual Studio Code"**

3. **Navigate to directories**  
   - Go to `/mount` or `/intern01_moss`

---

## To Mount NAS

Run the following scripts in the container:

```bash
./mountnas-all
./mountnas-workspace
```
