// This uart argent will have to act as the computer side of this pipeline
class uart_item extends uvm_sequence_item;
  `uvm_object_utils(uart_item);

    function new(string name = "uart_item");
        super.new(name);
    endfunction

    bit [7:0] data_byte;

endclass

// One PC -> FPGA frame:  [0xAA] [TYPE] [payload] [XOR checksum] [0x55]
//
// The test fills in type_byte and payload before calling start(), so this one
// class covers every frame the PC ever sends -- inference (0x01/0x02) and the
// eight weight matrices (0x10-0x17).
class uart_sequence extends uvm_sequence #(uart_item);
  `uvm_object_utils(uart_sequence)

    bit [7:0] type_byte = 8'h01;   // default: layer 1 inference
    bit [7:0] payload [];          // dynamic array -- the test sizes it

    function new(string name = "uart_sequence");
        super.new(name);
    endfunction

    // One uart_item carries exactly one byte, so every byte of the frame needs
    // its own start_item/finish_item pair. This wraps that.
    protected task send(bit [7:0] b);
        uart_item item;
        item = uart_item::type_id::create("item");
        start_item(item);
        item.data_byte = b;
        finish_item(item);          // returns once the driver has shifted it out
    endtask

    virtual task body();
        bit [7:0] checksum = 8'h00;

        send(8'hAA);                // start marker
        send(type_byte);

        foreach (payload[i]) begin
            send(payload[i]);
            checksum ^= payload[i]; // payload bytes only -- matches
        end                         // compute_checksum() in python_driver.py

        send(checksum);
        send(8'h55);                // stop marker
    endtask

endclass

// The sequencer needs no custom behaviour, so it is UVM's generic one told
// which transaction type it carries.
typedef uvm_sequencer #(uart_item) uart_sequencer;

// Turns uart_items into pin wiggles on vif.rx. Knows the baud timing and
// nothing about the frame protocol -- that lives in the sequences.
class uart_driver extends uvm_driver #(uart_item);
  `uvm_component_utils(uart_driver)

  // Handle to the interface instantiated in tb_top. "virtual" = pointer to an
  // interface; a class cannot hold a real one.
  virtual uart_if vif;

  // 100 MHz / 921600 baud = 108 clocks per bit, matching the CLKS_PER_BIT
  // localparam inside uart_rx.v.
  const int CLKS_PER_BIT = 108;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);

    // The matching half of the ::set() in tb_top. The "vif" string has to be
    // identical in both places or the lookup silently misses.
    if (!uvm_config_db #(virtual uart_if)::get(this, "", "vif", vif))
      `uvm_fatal(get_type_name(), "no virtual interface found under name 'vif'")
    endfunction

    virtual task run_phase(uvm_phase phase);
      vif.rx <= 1'b1;              // UART idles high
      wait (vif.rst == 1'b0);      // nothing to send while the DUT is in reset

      forever begin
        seq_item_port.get_next_item(req);   // blocks until a sequence sends one
        send_byte(req.data_byte);           // ~1080 clocks
        seq_item_port.item_done();          // release the sequencer
      end
    endtask

    // One UART character: start bit, 8 data bits LSB-first, stop bit.
    // Each bit is held for CLKS_PER_BIT clocks -- this is the only place in
    // the testbench that knows anything about baud rate.
    virtual task send_byte(bit [7:0] data);
      vif.rx <= 1'b0;                                  // start bit
      repeat (CLKS_PER_BIT) @(posedge vif.clk);

      for (int i = 0; i < 8; i++) begin                // data bits, LSB first
        vif.rx <= data[i];
        repeat (CLKS_PER_BIT) @(posedge vif.clk);
      end

      vif.rx <= 1'b1;                                  // stop bit
      repeat (CLKS_PER_BIT) @(posedge vif.clk);
    endtask


endclass


// Stub. Fills in once the driver is confirmed working on the waveform.
class uart_monitor extends uvm_monitor;
  `uvm_component_utils(uart_monitor)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

endclass


class uart_agent extends uvm_agent;
  `uvm_component_utils(uart_agent)

  uart_driver  driver;
  uart_sequencer sequencer;
  uart_monitor   monitor;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase); // Always call super first
    
    // Instantiate sub-components via factory
    driver = uart_driver::type_id::create("driver", this);
    sequencer = uart_sequencer::type_id::create("sequencer", this);
    monitor = uart_monitor::type_id::create("monitor", this);
  endfunction

  virtual function void connect_phase(uvm_phase phase);

    driver.seq_item_port.connect(sequencer.seq_item_export); //ask to explain this

  endfunction

endclass