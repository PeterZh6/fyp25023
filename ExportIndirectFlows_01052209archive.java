// @category Export
import java.io.*;
import java.util.List;
import java.util.ArrayList;
import java.util.TreeSet;

import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;
import ghidra.program.model.listing.Listing;
import ghidra.program.model.address.Address;
import ghidra.program.model.symbol.Reference;
import ghidra.program.model.symbol.RefType;
import ghidra.program.model.symbol.FlowType;



public class ExportIndirectFlows_01052209archive extends GhidraScript {

    private String funcName(Address a) {
        Function f = getFunctionContaining(a);
        return f == null ? "" : f.getName();
    }

    private List<String> collectTargets(Address from) {
        Reference[] refs = getReferencesFrom(from);
        ArrayList<String> computed = new ArrayList<>();
        ArrayList<String> flow = new ArrayList<>();
        for (Reference r : refs) {
            RefType t = r.getReferenceType();
            if (t == null || !t.isFlow()) continue;
            Address to = r.getToAddress();
            if (to == null) continue;
            String s = to.toString();
            flow.add(s);
            if (t.isComputed()) computed.add(s);
        }
        List<String> use = computed.isEmpty() ? flow : computed;
        // dedup + sort
        TreeSet<String> set = new TreeSet<>(use);
        return new ArrayList<>(set);
    }

    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        String outPath;
        if (args != null && args.length >= 1) outPath = args[0];
        else outPath = askFile("Save CSV", "Save").getAbsolutePath();

        Listing listing = currentProgram.getListing();
        InstructionIterator it = listing.getInstructions(true);

        int n = 0;
        try (PrintWriter w = new PrintWriter(new BufferedWriter(new FileWriter(outPath)))) {
            w.println("insn_addr,function,flow_type,insn_text,ghidra_targets,num_targets");
            while (it.hasNext()) {
                Instruction ins = it.next();
                FlowType ft = ins.getFlowType();

                if (ft == null || !ft.isComputed()) continue;
                if (ft.isTerminal()) continue;  // skip PLT stubs, i.e. eliminate all *_TERMINATOR types

                Address a = ins.getAddress();
                String fn = funcName(a);
                String text = ins.toString();
                List<String> targets = collectTargets(a);

                String joined = String.join(";", targets);
                w.printf("%s,%s,%s,%s,%s,%d%n",
                        a.toString(),
                        fn.replace(",", "_"),
                        ft.toString(),
                        text.replace(",", " "),
                        joined,
                        targets.size());
                n++;
            }
        }

        println("Wrote " + n + " rows to " + outPath);
    }
}
